import asyncio
import time
from tqdm import tqdm
import logging
import numpy as np
import torch
import tensorrt as trt
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

class CosyVoiceModel:

    def __init__(self, model_dir='', fp16=False, do_warmup=10):
        self.fp16 = fp16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flow_encoder = torch.jit.load('{}/fm_enc.ts'.format(model_dir), map_location=self.device)
        self.trt_runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        if self.fp16:
            with open('{}/fm_dec_sm80_fp16.plan'.format(model_dir), 'rb') as f:
                self.flow_decoder_trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())
                self.flow_decoder_trt_context = self.flow_decoder_trt_engine.create_execution_context()
        else:
            with open('{}/fm_dec_sm80_fp32.plan'.format(model_dir), 'rb') as f:
                self.flow_decoder_trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())
                self.flow_decoder_trt_context = self.flow_decoder_trt_engine.create_execution_context()
        self.generator = torch.jit.load('{}/hiftgen_24k.ts'.format(model_dir), map_location=self.device)
        self.f0_predictor = torch.jit.load('{}/f0predictor_24k.ts'.format(model_dir), map_location=self.device)
        self.embedding = torch.from_numpy(np.load('{}/xvec.npy'.format(model_dir))).to(self.device)
        self.min_decode_len = 6
        self.right_context = 4
        self.token_mel_ratio = 2
        self.hift_mel_cache_len = 8
        self.hift_source_cache_len = int(self.hift_mel_cache_len * 480)
        self.window = np.hamming(2 * self.hift_source_cache_len)
        self.mel_dim = 80
        self.noise = torch.randn((1, self.mel_dim, 4096), device=self.device)
        self.diff_steps = 10
        self.t_span = torch.linspace(0, 1, self.diff_steps + 1).to(self.device)
        self.t_span = 1 - torch.cos(self.t_span * 0.5 * torch.pi)
        self.cfg_rate = 0.7
        self.speech_token = torch.load('{}/speech_token.pt'.format(model_dir))
        for _ in tqdm(range(do_warmup), desc='cosyvoice warmup'):
            for speech in self.stream_inference(self.input_generator()):
                continue

    def input_generator(self):
        for i in range(0, len(self.speech_token), 10):
            yield self.speech_token[i: i + 10]

    @torch.inference_mode()
    def stream_inference(self, token_generator):
        tokens, token_offset, speech_cache, mel_cache, source_cache = [], 0, None, None, None
        for i in token_generator:
            tokens += i
            if len(tokens) - token_offset >= self.right_context + self.min_decode_len:
                mel = self.flow_inference(tokens, finalize=False)
                mel = mel[:, :, token_offset * self.token_mel_ratio:]
                speech, speech_cache, mel_cache, source_cache = self.hift_inference(mel, speech_cache, mel_cache, source_cache)
                # 开头有瑕疵，去掉20ms
                if token_offset == 0:
                    speech = speech[:, 480:]
                yield speech
                token_offset = len(tokens) - self.right_context
        # 处理剩余token
        mel = self.flow_inference(tokens, finalize=True)
        mel = mel[:, :, token_offset * self.token_mel_ratio:]
        speech, speech_cache, mel_cache, source_cache = self.hift_inference(mel, speech_cache, mel_cache, source_cache, finalize=True)
        yield speech

    @torch.inference_mode()
    def flow_inference(self, tokens, finalize=False):
        tokens = torch.tensor(tokens, device=self.device).unsqueeze(dim=0)
        token_lens = torch.tensor([tokens.shape[1]], device=self.device)
        mask_rand = torch.rand((tokens.shape[0], 1, 1), device=self.device, dtype=torch.float32)
        cal_len = token_lens * self.token_mel_ratio
        y = torch.zeros((1, cal_len[0], 80), device=self.device)

        # encoder
        h, mask, conds, rand_xvec = self.flow_encoder(tokens, token_lens, mask_rand, cal_len, self.embedding, y)
        # 如果不是尾包，回退context
        if not finalize:
            h = h[:, :, :-self.right_context * self.token_mel_ratio]
            mask, conds = mask[:, :, :-self.right_context * self.token_mel_ratio], conds[:, :, :-self.right_context * self.token_mel_ratio]

        # decoder
        t, dt = self.t_span[0], self.t_span[1] - self.t_span[0]
        x = self.noise[:, :, :h.size(2)]
        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        if self.fp16 is True:
            x = x.half()
        x_in_buffer = torch.zeros([2, 80, x.size(2)], device=self.device, dtype=x.dtype)
        mask_in_buffer = torch.zeros([2, 1, x.size(2)], device=self.device, dtype=x.dtype)
        mu_in_buffer = torch.zeros([2, 80, x.size(2)], device=self.device, dtype=x.dtype)
        t_in_buffer = torch.zeros([2], device=self.device, dtype=x.dtype)
        spks_in_buffer = torch.zeros([2, 80], device=self.device, dtype=x.dtype)
        cond_in_buffer = torch.zeros([2, 80, x.size(2)], device=self.device, dtype=x.dtype)
        steps = 1
        while steps <= len(self.t_span) - 1:
            x_in_buffer[0] = x
            x_in_buffer[1] = x
            mask_in_buffer[0] = mask
            mask_in_buffer[1] = mask
            mu_in_buffer[0] = h
            t_in_buffer[0] = t.unsqueeze(0)
            t_in_buffer[1] = t.unsqueeze(0)
            spks_in_buffer[0] = rand_xvec
            cond_in_buffer[0] = conds

            # prepare trt input
            if not x_in_buffer.is_contiguous():
                x_in_buffer = x_in_buffer.contiguous()
            if not mask_in_buffer.is_contiguous():
                mask_in_buffer = mask_in_buffer.contiguous()
            if not mu_in_buffer.is_contiguous():
                mu_in_buffer = mu_in_buffer.contiguous()
            if not t_in_buffer.is_contiguous():
                t_in_buffer = t_in_buffer.contiguous()
            if not spks_in_buffer.is_contiguous():
                spks_in_buffer = spks_in_buffer.contiguous()
            if not cond_in_buffer.is_contiguous():
                cond_in_buffer = cond_in_buffer.contiguous()
            if not mask_rand.is_contiguous():
                mask_rand = mask_rand.contiguous()
            self.flow_decoder_trt_context.set_input_shape('x', (2, 80, x.size(2)))
            self.flow_decoder_trt_context.set_input_shape('mask', (2, 1, x.size(2)))
            self.flow_decoder_trt_context.set_input_shape('mu', (2, 80, x.size(2)))
            self.flow_decoder_trt_context.set_input_shape('t', (2,))
            self.flow_decoder_trt_context.set_input_shape('spk', (2, 80))
            self.flow_decoder_trt_context.set_input_shape('cond', (2, 80, x.size(2)))
            self.flow_decoder_trt_context.set_input_shape('mask_rand', (1, 1, 1))

            # run trt engine
            self.flow_decoder_trt_context.execute_v2([x_in_buffer.data_ptr(),
                                                      mask_in_buffer.data_ptr(),
                                                      mu_in_buffer.data_ptr(),
                                                      t_in_buffer.data_ptr(),
                                                      spks_in_buffer.data_ptr(),
                                                      cond_in_buffer.data_ptr(),
                                                      mask_rand.data_ptr(),
                                                      x_in_buffer.data_ptr()])
            dphi_dt = x_in_buffer
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [1, 1], dim=0)
            dphi_dt = ((1.0 + self.cfg_rate) * dphi_dt - self.cfg_rate * cfg_dphi_dt)
            x = x + dphi_dt * dt
            t = t + dt

            if steps < len(self.t_span) - 1:
                dt = self.t_span[steps + 1] - t
            steps += 1
        return x.float()

    @torch.inference_mode()
    def hift_inference(self, mel, speech_cache, mel_cache, source_cache, finalize=False):
        f0 = self.f0_predictor(mel)
        if source_cache is None:
            assert mel_cache is None and speech_cache is None
            speech, source = self.generator(mel, f0)
        else:
            speech, source = self.generator(torch.concat([mel_cache, mel], dim=2), f0, source_cache)
            speech = self.cross_fade(speech, speech_cache)
        if not finalize:
            speech_cache = speech[:, :, -self.hift_source_cache_len:]
            mel_cache = mel[:, :, -self.hift_mel_cache_len:]
            source_cache = source[:, :, -self.hift_source_cache_len:]
            return speech[:, :, :-self.hift_source_cache_len].squeeze(dim=1), speech_cache, mel_cache, source_cache
        else:
            return speech.squeeze(dim=1), None, None, None

    @torch.inference_mode()
    def cross_fade(self, speech, speech_cache):
        speech, speech_cache = speech.cpu(), speech_cache.cpu()
        overlap_len = int(self.window.shape[0] / 2)
        if speech.device == torch.device('cpu'):
            speech = speech.clone()
        speech[..., :overlap_len] = speech[..., :overlap_len] * self.window[:overlap_len] + \
            speech_cache[..., -overlap_len:] * self.window[overlap_len:]
        return speech.to(self.device)

@torch.inference_mode()
async def async_stream_inference(model, token_queue):
    tokens, token_offset, speech_cache, mel_cache, source_cache = [], 0, None, None, None
    speech_len_list, time_list = [], []
    while True:
        if token_queue.empty() is False:
            i = token_queue.get()
            start_time = time.time()
            if i != []:
                tokens += i
                if len(tokens) - token_offset >= model.right_context + model.min_decode_len:
                    mel = model.flow_inference(tokens, finalize=False)
                    mel = mel[:, :, token_offset * model.token_mel_ratio:]
                    speech, speech_cache, mel_cache, source_cache = model.hift_inference(mel, speech_cache, mel_cache, source_cache)
                    # 开头有瑕疵，去掉20ms
                    if token_offset == 0:
                        speech = speech[:, 480:]
                    speech_len = speech.shape[1] / 24000
                    logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                    speech_len_list.append(speech_len)
                    time_list.append(time.time() - start_time)
                    start_time = time.time()
                    yield (speech.flatten() * (2**15)).cpu().numpy().astype(np.int16).tobytes()
                    token_offset = len(tokens) - model.right_context
            else:
                # 处理剩余token
                mel = model.flow_inference(tokens, finalize=True)
                mel = mel[:, :, token_offset * model.token_mel_ratio:]
                speech, speech_cache, mel_cache, source_cache = model.hift_inference(mel, speech_cache, mel_cache, source_cache, finalize=True)
                speech_len = speech.shape[1] / 24000
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                speech_len_list.append(speech_len)
                time_list.append(time.time() - start_time)
                start_time = time.time()
                yield (speech.flatten() * (2**15)).cpu().numpy().astype(np.int16).tobytes()
                yield b''
                break
        await asyncio.sleep(0.01)
    logging.info('average rtf {}'.format(sum(time_list) / sum(speech_len_list)))