_S3_ROOT = next((p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'env.sh').is_file() and (p / 'tools').is_dir()))
import os
import time
import torch
import torch.nn.functional as F
from functools import partial
from contextlib import nullcontext
import inspect
from logging import getLogger
logger = getLogger(__name__)
from typing import Union, Optional, List
import copy
from megatron.core.models.multimodal.fun_model import FunBaseModel
from megatron.core.models.multimodal.fun_model_channels import FunBaseModelChannels
from megatron.core.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_src_rank, get_tensor_model_parallel_group, get_tensor_model_parallel_rank
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.fun_asr_dataset import FunASRSFTDataset, FunASRDatasetConfig
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.arguments import core_transformer_config_from_args, core_audio_model_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec, get_gpt_layer_with_transformer_engine_spec, get_qwen3_5_moe_layer_specs
from megatron.core.models.multimodal.fun_spec import get_fun_audio_encoder_spec, get_fun_audio_adaptor_spec
from megatron.core.transformer.moe.flashmoe_router import batched_dropped_token_ratio, clear_num_dropped_tokens, get_load_balancing_loss, save_load_balancing_loss, clear_load_balancing_loss, batched_load_balancing_loss, clear_router_z_loss, batched_router_z_loss
from megatron.core.tensor_parallel import broadcast_data
from megatron.core.models.multimodal.fun_utils import max_seqlen_from_cu_seqlens
from funasr.register import tables
from megatron.core.models.multimodal.cosyvoice_compat import CosyVoiceMixedLMSimpleSysV2
import omegaconf
stimer = StragglerDetector()
_loss_func_debug_logged = False
_codec_text_tokenizer = None
_codec_text_retokenize_debug_logged = False
_cosyvoice_campplus_session = None

def _load_cosyvoice_native_init_checkpoint(model, args):
    """Initialize the native CosyVoice TTS branch and optional FM.

    TTS initialization is independent from FM construction so bridge/joint
    stages can avoid the expensive waveform/FM forward while still loading the
    pretrained Qwen2 codec LLM and codec embedding/head.
    """
    fm_checkpoint_path = os.environ.get('COSYVOICE_FM_INIT_CKPT', '').strip()
    tts_checkpoint_path = os.environ.get('COSYVOICE_TTS_INIT_CKPT', fm_checkpoint_path).strip()
    load_full_tts = os.environ.get('COSYVOICE_TTS_INIT_FULL', '0').strip().lower() in ('1', 'true', 'yes', 'on')
    checkpoint_path = tts_checkpoint_path if load_full_tts else fm_checkpoint_path
    if not checkpoint_path or not bool(getattr(args, 'finetune', False)):
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'CosyVoice native init checkpoint not found: {checkpoint_path}')
    print_rank_0(f'[MODEL INIT] loading native CosyVoice checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=True)
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f'CosyVoice FM checkpoint state_dict must be a dict, got {type(state_dict).__name__}')
    if load_full_tts:
        llm_prefix = 'llm_model.llm.model.'
        llm_state_dict = {name[len(llm_prefix):]: tensor for name, tensor in state_dict.items() if name.startswith(llm_prefix)}
        if not llm_state_dict:
            raise RuntimeError(f'No {llm_prefix}* tensors found in {checkpoint_path}')
        model.cosyvoice_llm.load_state_dict(llm_state_dict, strict=True)
        codec_embed_key = 'llm_model.codec_embed.weight'
        codec_head_key = 'llm_model.codec_head.weight'
        if codec_embed_key not in state_dict or codec_head_key not in state_dict:
            raise RuntimeError(f'Native TTS checkpoint is missing {codec_embed_key} or {codec_head_key}')
        native_embed = state_dict[codec_embed_key]
        target_embed = model.cosyvoice_codec_embed.weight
        if native_embed.shape[1:] != target_embed.shape[1:] or native_embed.shape[0] > target_embed.shape[0]:
            raise RuntimeError(f'codec embedding shape mismatch: native={tuple(native_embed.shape)}, target={tuple(target_embed.shape)}')
        with torch.no_grad():
            target_embed.zero_()
            target_embed[:native_embed.shape[0]].copy_(native_embed.to(target_embed.dtype))
        native_head = state_dict[codec_head_key]
        target_head = model.cosyvoice_llm_codec_head.weight
        tp_rank = mpu.get_tensor_model_parallel_rank()
        start = tp_rank * target_head.shape[0]
        end = min(start + target_head.shape[0], native_head.shape[0])
        with torch.no_grad():
            target_head.zero_()
            if end > start:
                target_head[:end - start].copy_(native_head[start:end].to(target_head.dtype))
        print_rank_0(f'[MODEL INIT] native TTS LLM/codec initialization passed: llm_tensors={len(llm_state_dict)}, codec_embed={tuple(native_embed.shape)}, codec_head={tuple(native_head.shape)}')
    fm_state_dict = None
    if getattr(model, 'has_cosyvoice_fm', False):
        if not fm_checkpoint_path:
            raise RuntimeError('FM is enabled but COSYVOICE_FM_INIT_CKPT is empty')
        if fm_checkpoint_path != checkpoint_path:
            fm_checkpoint = torch.load(fm_checkpoint_path, map_location='cpu', mmap=True, weights_only=True)
            fm_source = fm_checkpoint.get('state_dict', fm_checkpoint)
        else:
            fm_checkpoint = checkpoint
            fm_source = state_dict
        prefix = 'fm_model.'
        fm_state_dict = {name[len(prefix):]: tensor for name, tensor in fm_source.items() if name.startswith(prefix)}
        if not fm_state_dict:
            raise RuntimeError(f'No {prefix}* tensors found in {fm_checkpoint_path}')
        rotary_old = 'dit_model.rotary_embed.inv_freq'
        rotary_new = 'dit_model.rotary_embed.freqs'
        target_keys = model.cosyvoice_fm.state_dict().keys()
        if rotary_old in fm_state_dict and rotary_new in target_keys and (rotary_old not in target_keys):
            fm_state_dict[rotary_new] = fm_state_dict.pop(rotary_old)
            print_rank_0(f'[MODEL INIT] mapped pretrained FM rotary buffer {rotary_old} -> {rotary_new}')
        model.cosyvoice_fm.load_state_dict(fm_state_dict, strict=True)
        tensor_count = len(fm_state_dict)
        value_count = sum((tensor.numel() for tensor in fm_state_dict.values()))
        print_rank_0(f'[MODEL INIT] pretrained FM load passed: tensors={tensor_count}, values={value_count:,}, epoch={fm_checkpoint.get('epoch', '<unknown>')}, step={fm_checkpoint.get('step', '<unknown>')}')
        if fm_checkpoint_path != checkpoint_path:
            del fm_checkpoint
    del fm_state_dict, state_dict, checkpoint

def _configure_s2st_train_stage(model):
    """Apply adapter -> TTS LLM -> S2TT trainability for hidden-state S2ST."""
    stage = os.environ.get('S2ST_TRAIN_STAGE', '').strip().lower()
    if not stage:
        return
    aliases = {'bridge': 'adapter', 'joint': 'tts', 'full': 's2tt'}
    stage = aliases.get(stage, stage)
    if stage not in ('adapter', 'tts', 's2tt'):
        raise ValueError(f'S2ST_TRAIN_STAGE must be adapter, tts, or s2tt; got {stage!r}')
    for param in model.parameters():
        param.requires_grad = False
    for module_name in ('cosyvoice_codec_embed', 'cosyvoice_llm_codec_head'):
        module = getattr(model, module_name, None)
        if module is not None:
            module.train(stage in ('tts', 's2tt'))
            for param in module.parameters():
                param.requires_grad = stage in ('tts', 's2tt')
    bridge = getattr(model, 'cosyvoice_llm_input_proj', None)
    if bridge is None:
        raise RuntimeError('S2ST staged training requires cosyvoice_llm_input_proj')
    for param in bridge.parameters():
        param.requires_grad = True
    tts_llm = getattr(model, 'cosyvoice_llm', None)
    if tts_llm is None:
        raise RuntimeError('S2ST staged training requires cosyvoice_llm')
    if stage in ('tts', 's2tt'):
        tts_llm.train()
        for name, param in tts_llm.named_parameters():
            param.requires_grad = not name.startswith('embed_tokens.')
    else:
        tts_llm.eval()
    if stage == 's2tt':
        cosyvoice_prefixes = ('cosyvoice_llm.', 'cosyvoice_llm_input_proj.', 'cosyvoice_codec_embed.', 'cosyvoice_llm_codec_head.', 'cosyvoice_fm.')
        for name, param in model.named_parameters():
            if not name.startswith(cosyvoice_prefixes):
                param.requires_grad = True
    fm = getattr(model, 'cosyvoice_fm', None)
    if fm is not None:
        fm.eval()
        for param in fm.parameters():
            param.requires_grad = False
    by_branch = {}
    for name, param in model.named_parameters():
        branch = 'fm' if name.startswith('cosyvoice_fm.') else 'tts_llm' if name.startswith('cosyvoice_llm.') else 'bridge' if name.startswith('cosyvoice_llm_input_proj.') else 'codec_io' if name.startswith(('cosyvoice_codec_embed.', 'cosyvoice_llm_codec_head.')) else 's2tt'
        counts = by_branch.setdefault(branch, [0, 0])
        counts[0 if param.requires_grad else 1] += param.numel()
    expected_trainable = {'adapter': {'bridge'}, 'tts': {'bridge', 'tts_llm', 'codec_io'}, 's2tt': {'bridge', 'tts_llm', 'codec_io', 's2tt'}}[stage]
    for branch in ('bridge', 'tts_llm', 'codec_io', 'fm', 's2tt'):
        trainable = by_branch.get(branch, [0, 0])[0]
        should_train = branch in expected_trainable
        if should_train and trainable == 0:
            raise RuntimeError(f'S2ST stage {stage} expected trainable branch {branch}, but found none')
        if not should_train and trainable != 0:
            raise RuntimeError(f'S2ST stage {stage} unexpectedly left {trainable:,} trainable parameters in branch {branch}')
    print_rank_0(f'[S2ST STAGE] stage={stage}; ' + '; '.join((f'{name}:trainable={counts[0]:,},frozen={counts[1]:,}' for name, counts in sorted(by_branch.items()))))

class CodecTextRetokenizeError(ValueError):
    pass

def _use_cosyvoice_text_token_input(args):
    return bool(getattr(args, 'cosyvoice_codex_use_text_token_input', False) or os.environ.get('COSYVOICE_CODEX_USE_TEXT_TOKEN_INPUT', '0').strip().lower() in ('1', 'true', 'yes', 'on'))

def _get_codec_text_tokenizer(args):
    """Lazily load the codec LLM tokenizer in the data-producing process."""
    global _codec_text_tokenizer
    if _codec_text_tokenizer is not None:
        return _codec_text_tokenizer
    config_path = getattr(args, 'fun_dataset_config_path', None)
    if not config_path:
        raise ValueError('codec text-token input requires --fun-dataset-config-path')
    dataset_cfg = omegaconf.OmegaConf.load(config_path)
    llm_conf = dataset_cfg.get('model_conf', {}).get('llm_conf', {})
    codec_llm_conf = llm_conf.get('model_conf', {}).get('llm_conf', {})
    tokenizer_path = codec_llm_conf.get('tokenizer_init_param_path', None)
    if not tokenizer_path:
        tokenizer_path = codec_llm_conf.get('init_param_path', None)
    if not tokenizer_path:
        raise ValueError('codec text-token input requires llm_conf.init_param_path or llm_conf.tokenizer_init_param_path in the dataset YAML')
    from transformers import AutoTokenizer
    _codec_text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print_rank_0(f'Loaded codec text tokenizer from {tokenizer_path}')
    return _codec_text_tokenizer

def build_codec_text_input_ids(labels_ids, labels_ids_audio=None):
    """Retokenize supervised translation spans from main-tokenizer IDs to codec IDs.

    The returned tensor has the same packed layout as labels_ids. Each contiguous
    supervised span is decoded with the main tokenizer and encoded with the codec
    tokenizer. Remaining positions stay at -100 and contribute no text embedding.
    """
    global _codec_text_retokenize_debug_logged
    args = get_args()
    if not _use_cosyvoice_text_token_input(args):
        return None
    main_tokenizer = get_tokenizer()
    main_hf_tokenizer = getattr(main_tokenizer, '_tokenizer', main_tokenizer)
    codec_tokenizer = _get_codec_text_tokenizer(args)
    output = torch.full_like(labels_ids, -100)
    span_count = 0
    source_tokens = 0
    codec_tokens = 0
    truncated_tokens = 0
    for row_idx in range(labels_ids.shape[0]):
        valid_positions = torch.nonzero(labels_ids[row_idx] >= 0, as_tuple=False).view(-1)
        if valid_positions.numel() == 0:
            continue
        positions = valid_positions.tolist()
        spans = []
        span_start = positions[0]
        span_end = positions[0] + 1
        for position in positions[1:]:
            if position == span_end:
                span_end += 1
            else:
                spans.append((span_start, span_end))
                span_start, span_end = (position, position + 1)
        spans.append((span_start, span_end))
        for span_start, span_end in spans:
            if labels_ids_audio is not None and (not torch.any(labels_ids_audio[row_idx, span_start:span_end] >= 0)):
                continue
            main_ids = labels_ids[row_idx, span_start:span_end].tolist()
            text = main_hf_tokenizer.decode(main_ids, skip_special_tokens=True)
            if not text:
                continue
            ids = codec_tokenizer.encode(text, add_special_tokens=False)
            available = span_end - span_start
            used = min(len(ids), available)
            if used:
                output[row_idx, span_start:span_start + used] = torch.tensor(ids[:used], dtype=output.dtype, device=output.device)
            span_count += 1
            source_tokens += len(main_ids)
            codec_tokens += len(ids)
            truncated_tokens += max(0, len(ids) - available)
    if not _codec_text_retokenize_debug_logged:
        _codec_text_retokenize_debug_logged = True
        print_rank_0(f'[CODEX TEXT RETOKENIZE] spans={span_count}, main_tokens={source_tokens}, codec_tokens={codec_tokens}, truncated_tokens={truncated_tokens}')
    if truncated_tokens:
        raise CodecTextRetokenizeError(f'Codec-retokenized translation does not fit its aligned supervised span: truncated_tokens={truncated_tokens}. Refusing silent truncation.')
    return output

def _codex_batch_debug_enabled():
    return False

def _codex_batch_debug_threshold_s():
    return float(os.environ.get('CODEX_SLOW_BATCH_LOG_SEC', '5'))

def _codex_rank_info():
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else -1
    except Exception:
        rank = -1
    try:
        return {'rank': rank, 'tp_rank': mpu.get_tensor_model_parallel_rank(), 'dp_rank': mpu.get_data_parallel_rank(), 'pp_rank': mpu.get_pipeline_model_parallel_rank(), 'cp_rank': mpu.get_context_parallel_rank()}
    except Exception:
        return {'rank': rank}

def _codex_current_iteration():
    try:
        args = get_args()
        for name in ('curr_iteration', 'iteration'):
            value = getattr(args, name, None)
            if value is not None:
                return int(value)
    except Exception:
        pass
    return None

def _codex_tensor_stats(value):
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return {'shape': tuple(value.shape), 'dtype': str(value.dtype), 'empty': True}
            data = value.detach()
            if data.device.type != 'cpu':
                data = data.cpu()
            if not torch.is_floating_point(data):
                valid = data[data >= 0]
            else:
                valid = data
            if valid.numel() == 0:
                return {'shape': tuple(value.shape), 'dtype': str(value.dtype), 'valid': 0}
            return {'shape': tuple(value.shape), 'dtype': str(value.dtype), 'min': float(valid.min().item()), 'max': float(valid.max().item()), 'mean': float(valid.float().mean().item())}
        if isinstance(value, (list, tuple)):
            nums = []
            for item in value:
                if isinstance(item, torch.Tensor) and item.numel() > 0:
                    data = item.detach()
                    if data.device.type != 'cpu':
                        data = data.cpu()
                    nums.extend([float(x) for x in data.reshape(-1).tolist()])
                elif isinstance(item, (int, float)):
                    nums.append(float(item))
            if nums:
                return {'len': len(value), 'min': min(nums), 'max': max(nums), 'mean': sum(nums) / len(nums)}
            return {'len': len(value), 'type': type(value).__name__}
    except Exception as exc:
        return {'error': repr(exc), 'type': type(value).__name__}
    return {'type': type(value).__name__}

def _codex_batch_shape_summary(batch):
    if not isinstance(batch, dict):
        return {'type': type(batch).__name__}
    summary = {}
    for key in ('input_ids', 'labels_ids', 'input_ids_audio', 'labels_ids_audio', 'loss_mask', 'loss_mask_audio', 'attention_mask', 'position_ids', 'segment_fbank', 'segment_fbank_lengths', 'speech', 'speech_lengths', 'feature_beg', 'feature_type', 'fake_token_len', 'speech_token_id_offset', 'cu_seqlens'):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            summary[key] = {'shape': tuple(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)}
        elif value is None:
            summary[key] = None
        elif isinstance(value, (list, tuple)):
            summary[key] = {'len': len(value), 'type': type(value).__name__}
        else:
            summary[key] = {'type': type(value).__name__}
    sequence_chunks = batch.get('sequence_chunks')
    if isinstance(sequence_chunks, (list, tuple)):
        summary['sequence_chunks'] = sequence_chunks[:3]
    items = batch.get('items')
    if isinstance(items, (list, tuple)):
        summary['items_len'] = len(items)
    return summary

def _codex_summarize_items(items, limit=3):
    summary = []
    if not isinstance(items, (list, tuple)):
        return summary
    for item in items[:limit]:
        if isinstance(item, dict):
            entry = {'key': item.get('key'), 'source_len': item.get('source_len'), 'duration': item.get('duration')}
            user = item.get('user')
            if isinstance(user, (list, tuple)) and user:
                text = str(user[0])
                start = text.find('<|startofspeech|>')
                end = text.find('<|endofspeech|>')
                if start >= 0 and end > start:
                    audio_path = text[start + len('<|startofspeech|>'):end]
                    entry['audio'] = audio_path[:180]
                entry['user_prefix'] = text[:120]
            assistant = item.get('assistant')
            if isinstance(assistant, (list, tuple)) and assistant:
                entry['assistant_len'] = len(str(assistant[0]))
            summary.append(entry)
        else:
            summary.append({'type': type(item).__name__, 'repr': repr(item)[:240]})
    return summary

def _codex_data_summary(data):
    if not isinstance(data, dict):
        return {'type': type(data).__name__}
    summary = {'type': 'dict', 'keys': list(data.keys())[:30]}
    for key in ('input_ids', 'labels_ids', 'input_ids_audio', 'labels_ids_audio', 'attention_mask', 'segment_fbank', 'segment_fbank_lengths', 'speech', 'speech_lengths', 'feature_beg', 'feature_type', 'fake_token_len', 'cu_seqlens'):
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            summary[key] = {'shape': tuple(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)}
        elif isinstance(value, (list, tuple)):
            summary[key] = {'len': len(value), 'type': type(value).__name__}
        elif value is not None:
            summary[key] = {'type': type(value).__name__}
    for item_key in ('item', 'items'):
        if item_key in data:
            summary[item_key] = _codex_summarize_items(data.get(item_key))
    return summary

def _codex_log_batch_phase(phase, elapsed_s, batch=None, extra=None, force=False):
    if not _codex_batch_debug_enabled():
        return
    if not force and elapsed_s < _codex_batch_debug_threshold_s():
        return
    payload = _codex_rank_info()
    payload.update({'iteration': _codex_current_iteration(), 'phase': phase, 'elapsed_s': round(elapsed_s, 3)})
    if extra:
        payload.update(extra)
    if batch is not None:
        payload['batch_shapes'] = _codex_batch_shape_summary(batch)
        payload['items'] = _codex_summarize_items(batch.get('items'))
    print(f'[CODEX_BATCH_DEBUG] {payload}', flush=True)

def _codex_log_slow_batch(batch, elapsed_s):
    if not _codex_batch_debug_enabled():
        return
    if elapsed_s < _codex_batch_debug_threshold_s():
        return
    if batch is None:
        print(f"[CODEX_SLOW_BATCH] {{'elapsed_s': {round(elapsed_s, 3)}, 'batch': None}}", flush=True)
        return
    summary = _codex_rank_info()
    summary.update({'iteration': _codex_current_iteration(), 'elapsed_s': round(elapsed_s, 3), 'input_ids': _codex_tensor_stats(batch.get('input_ids')), 'labels_ids': _codex_tensor_stats(batch.get('labels_ids')), 'input_ids_audio': _codex_tensor_stats(batch.get('input_ids_audio')), 'speech': _codex_tensor_stats(batch.get('speech')), 'speech_lengths': _codex_tensor_stats(batch.get('speech_lengths')), 'fake_token_len': _codex_tensor_stats(batch.get('fake_token_len')), 'feature_beg': _codex_tensor_stats(batch.get('feature_beg')), 'items': _codex_summarize_items(batch.get('items'))})
    print(f'[CODEX_SLOW_BATCH] {summary}', flush=True)

def _codex_log_iterator_phase(phase, elapsed_s, data=None, extra=None, force=False):
    if not _codex_batch_debug_enabled():
        return
    if not force and elapsed_s < _codex_batch_debug_threshold_s():
        return
    payload = _codex_rank_info()
    payload.update({'iteration': _codex_current_iteration(), 'phase': phase, 'elapsed_s': round(elapsed_s, 3)})
    if extra:
        payload.update(extra)
    if data is not None:
        payload['data_summary'] = _codex_data_summary(data)
    print(f'[CODEX_ITERATOR_DEBUG] {payload}', flush=True)
if torch.__version__ < '2.5.0':
    print(f'WARNING: PyTorch version is {torch.__version__}, which is less than 2.5.0. Trying to bypass Triton Error temporarily by using torch.backward()')
    torch.empty(1, device='cuda', requires_grad=True).backward()

def find_last_router(model):
    last_router = None
    for name, module in model.named_children():
        if hasattr(module, 'is_last_router'):
            last_router = module
        last_router_in_submodule = find_last_router(module)
        if last_router_in_submodule is not None:
            last_router = last_router_in_submodule
    return last_router

def set_encoder_args_qwen3_common(encoder_args):
    encoder_args.num_experts = None
    encoder_args.use_flash_moe = False
    encoder_args.use_flash_moe_hep = False
    encoder_args.use_flash_moe_router = False
    encoder_args.moe_grouped_gemm = False
    encoder_args.bf16 = True
    encoder_args.sequence_parallel = False
    encoder_args.tensor_model_parallel_size = 1
    encoder_args.pipeline_model_parallel_size = 1
    encoder_args.decoder_first_pipeline_num_layers = None
    encoder_args.decoder_last_pipeline_num_layers = None
    encoder_args.context_parallel_size = 1
    encoder_args.expert_model_parallel_size = 1
    if encoder_args.audio_recompute_encoder:
        encoder_args.recompute_granularity = 'full'
        encoder_args.recompute_method = 'block'
        encoder_args.recompute_num_layers = encoder_args.num_layers

def encoder_args_qwen3_0_6B(args):
    encoder_args = copy.deepcopy(args)
    encoder_args.multi_latent_attention = False
    encoder_args.diff_attention = False
    encoder_args.rotary_base = 1000000
    encoder_args.qk_layernorm = True
    encoder_args.ffn_hidden_size = 3072
    encoder_args.group_query_attention = True
    encoder_args.kv_channels = 128
    encoder_args.num_query_groups = 8
    encoder_args.disable_bias_linear = True
    encoder_args.use_rotary_position_embeddings = True
    encoder_args.no_position_embedding = True
    encoder_args.hidden_dropout = 0
    encoder_args.attention_dropout = 0
    encoder_args.use_flash_attn = True
    encoder_args.swiglu = True
    encoder_args.normalization = 'RMSNorm'
    encoder_args.init_method_std = 0.009
    encoder_args.num_layers = 28
    encoder_args.hidden_size = 1024
    encoder_args.num_attention_heads = 16
    encoder_args.seq_length = 8192
    encoder_args.max_position_embeddings = 32768
    encoder_args.norm_epsilon = 1e-06
    set_encoder_args_qwen3_common(encoder_args)
    return encoder_args

def encoder_args_qwen3_2B(args):
    encoder_args = copy.deepcopy(args)
    encoder_args.multi_latent_attention = False
    encoder_args.diff_attention = False
    encoder_args.rotary_base = 1000000
    encoder_args.qk_layernorm = True
    encoder_args.ffn_hidden_size = 6144
    encoder_args.group_query_attention = True
    encoder_args.kv_channels = 128
    encoder_args.num_query_groups = 8
    encoder_args.disable_bias_linear = True
    encoder_args.use_rotary_position_embeddings = True
    encoder_args.no_position_embedding = True
    encoder_args.hidden_dropout = 0
    encoder_args.attention_dropout = 0
    encoder_args.use_flash_attn = True
    encoder_args.swiglu = True
    encoder_args.normalization = 'RMSNorm'
    encoder_args.init_method_std = 0.009
    encoder_args.num_layers = 40
    encoder_args.hidden_size = 2048
    encoder_args.num_attention_heads = 16
    encoder_args.seq_length = 8192
    encoder_args.max_position_embeddings = 32768
    encoder_args.norm_epsilon = 1e-06
    set_encoder_args_qwen3_common(encoder_args)
    return encoder_args

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    use_te = args.transformer_impl == 'transformer_engine'
    print_rank_0('building FunBase model ...')
    config = core_transformer_config_from_args(args)
    audio_config = core_audio_model_config_from_args(args)
    configure_cosyvoice_codex_from_dataset_config(args, audio_config)
    if args.audio_encoder_type.startswith('qwen3'):
        encoder_args = copy.deepcopy(args)
        if args.audio_encoder_type == 'qwen3-0.6b':
            encoder_args = encoder_args_qwen3_0_6B(encoder_args)
        elif args.audio_encoder_type == 'qwen3-2b':
            encoder_args = encoder_args_qwen3_2B(encoder_args)
        else:
            raise ValueError(f'Unsupported audio encoder type: {args.audio_encoder_type}')
        audio_encoder_transformer_config = core_transformer_config_from_args(encoder_args)
        audio_config.encoder_transformer_config = audio_encoder_transformer_config
    if args.use_legacy_models:
        raise NotImplementedError('Legacy models not supported for FunBase')
    else:
        if config.layer_types is not None:
            from megatron.core.transformer.transformer_block import get_num_layers_to_build
            num_layers_this_stage = get_num_layers_to_build(config)
            layer_offset = 0
            for pp_rank in range(mpu.get_pipeline_model_parallel_rank()):
                layer_offset += get_num_layers_to_build(config, pp_rank)
            stage_layer_types = config.layer_types[layer_offset:layer_offset + num_layers_this_stage]
            global_rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
            print(f'rank {global_rank}, PP rank {mpu.get_pipeline_model_parallel_rank()}, num_layers_this_stage: {num_layers_this_stage}, layer_offset: {layer_offset}, stage_layer_types: {stage_layer_types}')
            transformer_layer_spec = get_qwen3_5_moe_layer_specs(num_experts=args.num_experts, moe_grouped_gemm=args.moe_grouped_gemm, qk_layernorm=args.qk_layernorm, flash_moe=args.use_flash_moe, fp8=args.fp8 if use_te else None, layer_types=stage_layer_types, chunkflow=args.use_chunkflow, attn_output_gate=config.attn_output_gate)
            audio_encoder_spec = get_fun_audio_encoder_spec(use_te=use_te)
            audio_encoder_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(num_experts=None, moe_grouped_gemm=False, qk_layernorm=True, multi_latent_attention=False, diff_attention=False, flash_moe=False, fp8=args.fp8, chunkflow=False, causal=False)
        elif use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention, args.diff_attention, args.use_flash_moe, args.fp8, args.use_chunkflow)
            audio_encoder_spec = get_fun_audio_encoder_spec(use_te=True)
            audio_encoder_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(num_experts=None, moe_grouped_gemm=False, qk_layernorm=True, multi_latent_attention=False, diff_attention=False, flash_moe=False, fp8=args.fp8, chunkflow=False, causal=False)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention, args.use_diff_attention, args.use_flash_moe)
            audio_encoder_spec = get_fun_audio_encoder_spec(use_te=False)
            audio_encoder_transformer_layer_spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=False, qk_layernorm=True, multi_latent_attention=False, diff_attention=False, flash_moe=False, causal=False)
        audio_adaptor_spec = get_fun_audio_adaptor_spec(use_te)
        audio_config.encoder_transformer_layer_spec = audio_encoder_transformer_layer_spec
        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init
                build_model_context = fp8_model_init
                build_model_context_args['enabled'] = True
                if 'preserve_high_precision_init_val' in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args['preserve_high_precision_init_val'] = True
            except:
                raise RuntimeError('--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.')
        model_class = FunBaseModelChannels if audio_config.is_dual_channel else FunBaseModel
        print(f'Using model class: {model_class.__name__}')
        with build_model_context(**build_model_context_args):
            model = model_class(config=config, audio_config=audio_config, transformer_layer_spec=transformer_layer_spec, vocab_size=args.padded_vocab_size, max_sequence_length=args.max_position_embeddings, seq_length=args.seq_length, pre_process=pre_process, post_process=post_process, fp16_lm_cross_entropy=args.fp16_lm_cross_entropy, parallel_output=True, share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights, position_embedding_type=args.position_embedding_type, rotary_percent=args.rotary_percent, rotary_base=args.rotary_base, rope_scaling=args.use_rope_scaling, seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor, length_normalized_loss=getattr(args, 'length_normalized_loss', False), audio_encoder_activation_checkpoint=getattr(args, 'audio_encoder_activation_checkpoint', False), feat_permute=getattr(args, 'feat_permute', True))
    if getattr(args, 'lora_rank', 0) > 0:
        from megatron.core.transformer.lora import inject_lora_into_model
        inject_lora_into_model(model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=getattr(args, 'lora_dropout', 0.0), skip_linear_attention_out_proj=getattr(args, 'lora_skip_linear_attention_out_proj', False))
        if getattr(args, 'lora_freeze', False):
            frozen_lora_params = 0
            frozen_lora_tensors = 0
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = False
                    frozen_lora_params += param.numel()
                    frozen_lora_tensors += 1
            print_rank_0(f'[LoRA] Frozen adapters: tensors={frozen_lora_tensors}, parameters={frozen_lora_params:,}')
    _load_cosyvoice_native_init_checkpoint(model, args)
    _configure_s2st_train_stage(model)
    if getattr(model, 'has_cosyvoice_codex_head', False) and float(getattr(args, 'cosyvoice_codex_loss_weight', 1.0)) == 0.0 and (not getattr(model, 'has_cosyvoice_fm', False) or bool(getattr(args, 'cosyvoice_fm_only', False))):
        frozen_codec_tensors = 0
        frozen_codec_params = 0
        for module_name in ('cosyvoice_llm', 'cosyvoice_llm_input_proj', 'cosyvoice_codec_embed', 'cosyvoice_llm_codec_head'):
            module = getattr(model, module_name, None)
            if module is None:
                continue
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    frozen_codec_tensors += 1
                    frozen_codec_params += param.numel()
                    param.requires_grad = False
        print_rank_0(f'[MODEL INIT] froze CosyVoice codex branch because cosyvoice_codex_loss_weight=0.0, fm_only={bool(getattr(args, 'cosyvoice_fm_only', False))}: tensors={frozen_codec_tensors}, parameters={frozen_codec_params:,}')
    if bool(getattr(args, 'cosyvoice_fm_only', False)):
        unexpected_trainable = [name for name, param in model.named_parameters() if param.requires_grad and (not name.startswith('cosyvoice_fm.'))]
        if unexpected_trainable:
            preview = ', '.join(unexpected_trainable[:20])
            raise RuntimeError(f'COSYVOICE_FM_ONLY=1 left non-FM parameters trainable: {preview}')
        fm_trainable_params = sum((param.numel() for name, param in model.named_parameters() if param.requires_grad and name.startswith('cosyvoice_fm.')))
        print_rank_0(f'[MODEL INIT] FM-only trainable parameter check passed: parameters={fm_trainable_params:,}')
    return model

def configure_cosyvoice_codex_from_dataset_config(args, audio_config):
    """Attach optional CosyVoice codex-head settings from the FunASR YAML config."""
    audio_config.cosyvoice_codex_enabled = False
    audio_config.cosyvoice_fm_enabled = False
    audio_config.cosyvoice_fm_model_conf = {}
    args.cosyvoice_fm_only = False
    audio_config.cosyvoice_codex_codec_unit = None
    audio_config.cosyvoice_codex_model_name = None
    audio_config.cosyvoice_codex_llm_init_param_path = None
    audio_config.cosyvoice_codex_llm_dtype = 'fp32'
    audio_config.cosyvoice_codex_llm_freeze = True
    audio_config.cosyvoice_codex_use_text_token_input = _use_cosyvoice_text_token_input(args)
    audio_config.cosyvoice_codex_vocab_padding_multiple = int(os.environ.get('COSYVOICE_CODEX_VOCAB_PADDING_MULTIPLE', '0') or 0)
    audio_config.audio_loss_weight = 1.0
    args.cosyvoice_codex_loss_weight = 1.0
    args.audio_loss_weight = 1.0
    args.text_loss_weight = 1.0
    config_path = getattr(args, 'fun_dataset_config_path', None)
    if not config_path:
        return
    try:
        dataset_cfg = omegaconf.OmegaConf.load(config_path)
    except Exception as exc:
        print_rank_0(f'Skip CosyVoice codex config: failed to load {config_path}: {exc}')
        return
    llm_conf = dataset_cfg.get('model_conf', {}).get('llm_conf', {})
    model_name = llm_conf.get('model', None)
    enabled = bool(llm_conf.get('enabled', False))
    if model_name != 'CosyVoiceMixedLMSimpleSysV2' or not enabled:
        return
    model_conf = llm_conf.get('model_conf', {})
    cosyvoice_llm_conf = model_conf.get('llm_conf', {})
    speech_token_extractor_conf = cosyvoice_llm_conf.get('speech_token_extractor_conf', None)
    if speech_token_extractor_conf is not None:
        speech_token_extractor_conf = omegaconf.OmegaConf.to_container(speech_token_extractor_conf, resolve=True)
        if 'downsample_factor' not in speech_token_extractor_conf:
            tokenizer_frame_rate = float(getattr(args, 'audio_tokenizer_frame_rate', 25))
            speech_token_extractor_conf['downsample_factor'] = 8 if tokenizer_frame_rate == 12.5 else 4
        audio_config.speech_token_extractor_conf = speech_token_extractor_conf
        automodel_conf = speech_token_extractor_conf.get('automodel_conf', {})
        print_rank_0(f'Use speech_token_extractor_conf from FunASR YAML for main SpeechTokenizer: init_param={automodel_conf.get('init_param', None)}')
    codec_unit = int(model_conf.get('codec_unit', audio_config.tokenizer_code_book_size))
    loss_weight = float(os.environ.get('COSYVOICE_CODEX_LOSS_WEIGHT', llm_conf.get('loss_weight', 1.0)))
    audio_loss_weight = float(llm_conf.get('audio_loss_weight', 1.0))
    text_loss_weight = float(os.environ.get('TEXT_LOSS_WEIGHT', llm_conf.get('text_loss_weight', 1.0)))
    audio_config.cosyvoice_codex_enabled = True
    audio_config.cosyvoice_codex_codec_unit = codec_unit
    audio_config.cosyvoice_codex_model_name = model_name
    audio_config.cosyvoice_codex_llm_init_param_path = cosyvoice_llm_conf.get('init_param_path', None)
    audio_config.cosyvoice_codex_llm_dtype = cosyvoice_llm_conf.get('llm_dtype', 'fp32')
    audio_config.cosyvoice_codex_llm_freeze = bool(cosyvoice_llm_conf.get('freeze', True))
    if audio_config.cosyvoice_codex_vocab_padding_multiple <= 0:
        audio_config.cosyvoice_codex_vocab_padding_multiple = int(os.environ.get('COSYVOICE_CODEX_VOCAB_PADDING_MULTIPLE', '0') or 0)
    audio_config.audio_loss_weight = audio_loss_weight
    args.cosyvoice_codex_loss_weight = loss_weight
    args.audio_loss_weight = audio_loss_weight
    args.text_loss_weight = text_loss_weight
    fm_enabled = os.environ.get('ENABLE_COSYVOICE_FM', '0').strip().lower() in ('1', 'true', 'yes', 'on')
    args.cosyvoice_fm_only = os.environ.get('COSYVOICE_FM_ONLY', '0').strip().lower() in ('1', 'true', 'yes', 'on')
    if args.cosyvoice_fm_only and (not fm_enabled):
        raise ValueError('COSYVOICE_FM_ONLY=1 requires ENABLE_COSYVOICE_FM=1')
    if fm_enabled:
        if int(getattr(args, 'pipeline_model_parallel_size', 1)) != 1:
            raise ValueError('Joint CosyVoice FM currently requires pipeline_model_parallel_size=1')
        if int(getattr(args, 'context_parallel_size', 1)) != 1:
            raise ValueError('Joint CosyVoice FM currently requires context_parallel_size=1')
        if int(getattr(args, 'tensor_model_parallel_size', 1)) > 1 and (not bool(getattr(args, 'sequence_parallel', False))):
            raise ValueError('Joint CosyVoice FM with TP>1 requires --sequence-parallel')
        fm_conf = dataset_cfg.get('model_conf', {}).get('fm_conf', {})
        if fm_conf.get('model', None) != 'CosyVoiceFlowMatching':
            raise ValueError('ENABLE_COSYVOICE_FM=1 requires model_conf.fm_conf.model=CosyVoiceFlowMatching')
        audio_config.cosyvoice_fm_enabled = True
        audio_config.cosyvoice_fm_model_conf = omegaconf.OmegaConf.to_container(fm_conf.get('model_conf', {}), resolve=True)
    args.cosyvoice_fm_loss_weight = float(os.environ.get('COSYVOICE_FM_LOSS_WEIGHT', '1.0'))
    print_rank_0(f'Enable CosyVoiceMixedLMSimpleSysV2 codex auxiliary head: codec_unit={codec_unit}, loss_weight={loss_weight}, audio_loss_weight={audio_loss_weight}, text_loss_weight={text_loss_weight}, llm_init={audio_config.cosyvoice_codex_llm_init_param_path}, llm_dtype={audio_config.cosyvoice_codex_llm_dtype}, llm_freeze={audio_config.cosyvoice_codex_llm_freeze}, use_text_token_input={audio_config.cosyvoice_codex_use_text_token_input}')
    if fm_enabled:
        print_rank_0(f'Enable joint codec-LLM -> CosyVoice FM training: loss_weight={args.cosyvoice_fm_loss_weight}, fm_only={args.cosyvoice_fm_only}')

def scale_lr_cond(name: str, param: torch.nn.Parameter) -> bool:
    return getattr(param, 'lr_scale', False)

def broadcast_anything(item: Union[torch.Tensor, list, dict]):
    if mpu.get_tensor_model_parallel_rank() == 0:
        if isinstance(item, torch.Tensor):
            payload = {'object': None, 'object_type': type(item), 'shape': list(item.shape), 'dtype': item.dtype, 'device': item.device.type}
        else:
            payload = {'object': item, 'object_type': type(item), 'shape': None, 'dtype': None, 'device': None}
    else:
        payload = None
    object_list = [payload]
    torch.distributed.broadcast_object_list(object_list, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
    payload = object_list[0]
    if payload['object_type'] == torch.Tensor:
        if mpu.get_tensor_model_parallel_rank() == 0:
            if item.device.type == 'cpu':
                to_broadcast = item.cuda(non_blocking=True)
            else:
                to_broadcast = item
            torch.distributed.broadcast(to_broadcast, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
        else:
            to_broadcast = torch.empty(payload['shape'], dtype=payload['dtype'], device='cuda')
            torch.distributed.broadcast(to_broadcast, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
            item = to_broadcast
            if payload['device'] == 'cpu':
                item = item.cpu()
    elif mpu.get_tensor_model_parallel_rank() != 0:
        item = payload['object']
    return item
AUDIO_BATCH_KEYS = ['segment_fbank', 'segment_fbank_lengths', 'speech', 'speech_lengths', 'feature_beg', 'feature_type', 'fake_token_len', 'items', 'speech_token_id_offset', 'sequence_chunks', 'target_audio', 'target_audio_lengths', 'target_spk_embedding']

def _extract_cosyvoice_spk_embedding(wav_16k):
    """Reference CAM++ speaker embedding: 16 kHz fbank -> 192-D ONNX output."""
    global _cosyvoice_campplus_session
    import onnxruntime
    import torchaudio.compliance.kaldi as kaldi
    if _cosyvoice_campplus_session is None:
        model_path = os.environ.get('COSYVOICE_SPK_EMBED_MODEL', '__S3_ROOT__/models/speaker/campplus.onnx'.replace('__S3_ROOT__', str(_S3_ROOT)))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'CAM++ speaker embedding model not found: {model_path}')
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        _cosyvoice_campplus_session = onnxruntime.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
    wav_16k = torch.as_tensor(wav_16k, dtype=torch.float32).reshape(1, -1)
    max_len = 10 * 16000
    if wav_16k.shape[1] > max_len:
        start = int(torch.randint(0, wav_16k.shape[1] - max_len + 1, ()).item())
        wav_16k = wav_16k[:, start:start + max_len]
    feat = kaldi.fbank(wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    session = _cosyvoice_campplus_session
    embedding = session.run(None, {session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()})[0]
    return torch.from_numpy(embedding).float().reshape(192)

def _extract_fm_target_audio_paths(item, pattern):
    """Extract target audio paths from both legacy and messages-style samples."""
    if not isinstance(item, dict):
        return []
    values = []
    assistant = item.get('assistant', [])
    if not isinstance(assistant, (list, tuple)):
        assistant = [assistant]
    values.extend(assistant)
    messages = item.get('messages', [])
    if isinstance(messages, (list, tuple)):
        for message in messages:
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                continue
            for key in ('audio', 'target_audio'):
                value = message.get(key)
                if value:
                    values.append(value)
    paths = []
    for value in values:
        paths.extend(pattern.findall(str(value)))
    return paths

def _load_fm_target_audio(items):
    """Load assistant audio at 24 kHz and align every turn to one codec frame."""
    if os.environ.get('ENABLE_COSYVOICE_FM', '0').strip().lower() not in ('1', 'true', 'yes', 'on'):
        return (None, None, None)
    from funasr.datasets.minmo_datasets.datasets import _rewrite_audio_path_for_oss_cache
    from funasr.utils.load_utils import load_audio_text_image_video
    import torchaudio
    import re
    sample_rate = int(os.environ.get('COSYVOICE_FM_SAMPLE_RATE', '24000'))
    if sample_rate != 24000:
        raise ValueError(f'The configured CosyVoice FM uses 24 kHz mel features; COSYVOICE_FM_SAMPLE_RATE must be 24000, got {sample_rate}')
    samples_per_codec = 1920
    pattern = re.compile('<\\|startofaudiotoken\\|>!?(.+?)<\\|endofaudiotoken\\|>')
    packed_items = items if isinstance(items, list) else [items]
    waves = []
    speaker_embeddings = []
    for item_index, item in enumerate(packed_items):
        paths = _extract_fm_target_audio_paths(item, pattern)
        if not paths:
            keys = sorted(item.keys()) if isinstance(item, dict) else []
            raise ValueError(f'FM is enabled but packed sample item {item_index} has no assistant target audio (keys={keys})')
        parts = []
        last_wav_16k = None
        for path in paths:
            if os.path.isfile(path):
                wav_native, source_rate = torchaudio.load(path)
                wav_native = wav_native.sum(dim=0, keepdim=True)
                wav_16k = torchaudio.functional.resample(wav_native, orig_freq=source_rate, new_freq=16000) if source_rate != 16000 else wav_native
                wav = torchaudio.functional.resample(wav_16k, orig_freq=16000, new_freq=sample_rate).reshape(-1)
            else:
                path_for_load = _rewrite_audio_path_for_oss_cache(path, use_oss_cache=True)
                wav_16k = load_audio_text_image_video(path_for_load, fs=16000, use_ffmpeg=True)
                wav_16k = torch.as_tensor(wav_16k, dtype=torch.float32).reshape(1, -1)
                wav = torchaudio.functional.resample(wav_16k, orig_freq=16000, new_freq=sample_rate).reshape(-1)
            last_wav_16k = wav_16k
            pad_size = -wav.numel() % samples_per_codec
            if pad_size:
                wav = F.pad(wav, (0, pad_size))
            parts.append(wav)
        waves.append(torch.cat(parts))
        speaker_embeddings.append(_extract_cosyvoice_spk_embedding(last_wav_16k))
    lengths = torch.tensor([wav.numel() for wav in waves], dtype=torch.long)
    return (torch.nn.utils.rnn.pad_sequence(waves, batch_first=True), lengths, torch.stack(speaker_embeddings))

def get_batch_on_this_tp_rank(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    total_start = time.time()
    if tp_rank == 0:
        if data_iterator is not None:
            next_start = time.time()
            data = next(data_iterator)
            _codex_log_batch_phase('tp0_next_data_iterator', time.time() - next_start, data if isinstance(data, dict) else None, extra={'data_type': type(data).__name__})
            copy_start = time.time()
            target_audio, target_audio_lengths, target_spk_embedding = _load_fm_target_audio(data.get('items'))
            if data.get('input_ids_audio', None) is not None:
                input_ids_audio = data['input_ids_audio'].cuda(non_blocking=True)
                labels_ids_audio = data['labels_ids_audio'].cuda(non_blocking=True)
                loss_mask_audio = data['loss_mask_audio'].cuda(non_blocking=True)
            else:
                input_ids_audio = None
                labels_ids_audio = None
                loss_mask_audio = None
            batch = {'input_ids': data['input_ids'].cuda(non_blocking=True), 'labels_ids': data['labels_ids'].cuda(non_blocking=True), 'input_ids_codec_text': data['input_ids_codec_text'].cuda(non_blocking=True) if data.get('input_ids_codec_text', None) is not None else None, 'input_ids_audio': input_ids_audio, 'labels_ids_audio': labels_ids_audio, 'loss_mask': data['loss_mask'].cuda(non_blocking=True), 'loss_mask_audio': loss_mask_audio, 'attention_mask': data['attention_mask'].cuda(non_blocking=True), 'position_ids': data['position_ids'].cuda(non_blocking=True), 'segment_fbank': data['segment_fbank'].cuda(non_blocking=True) if data.get('segment_fbank', None) is not None else None, 'segment_fbank_lengths': data['segment_fbank_lengths'], 'speech': data['speech'].cuda(non_blocking=True) if data.get('speech', None) is not None else None, 'speech_lengths': data['speech_lengths'], 'speech_info': data['speech_info'], 'feature_beg': data['feature_beg'], 'feature_type': data['feature_type'], 'fake_token_len': data['fake_token_len'], 'items': data['items'], 'speech_token_id_offset': data['speech_token_id_offset'], 'sequence_chunks': data['sequence_chunks'], 'target_audio': target_audio.cuda(non_blocking=True) if target_audio is not None else None, 'target_audio_lengths': target_audio_lengths.cuda(non_blocking=True) if target_audio_lengths is not None else None, 'target_spk_embedding': target_spk_embedding.cuda(non_blocking=True) if target_spk_embedding is not None else None}
            if 'cu_seqlens' in data:
                batch['cu_seqlens'] = data['cu_seqlens'].cuda(non_blocking=True) if isinstance(data['cu_seqlens'], torch.Tensor) else data['cu_seqlens']
            _codex_log_batch_phase('tp0_cuda_copy_and_batch_build', time.time() - copy_start, batch)
            broadcast_start = time.time()
            broadcast_anything(batch['input_ids'])
            broadcast_anything(batch['input_ids_audio'])
            broadcast_anything(batch['labels_ids'])
            broadcast_anything(batch['input_ids_codec_text'])
            broadcast_anything(batch['labels_ids_audio'])
            broadcast_anything(batch['loss_mask'])
            broadcast_anything(batch['loss_mask_audio'])
            broadcast_anything(batch['attention_mask'])
            broadcast_anything(batch['position_ids'])
            for key in AUDIO_BATCH_KEYS:
                broadcast_anything(batch[key])
            if 'cu_seqlens' in batch:
                broadcast_anything(batch['cu_seqlens'])
            _codex_log_batch_phase('tp0_broadcast_batch', time.time() - broadcast_start, batch)
        else:
            data = None
            batch = {}
        if args.use_chunkflow or args.use_smart_batching:
            raise NotImplementedError('ChunkFlow and SmartBatching are not supported for FunBase')
        elif args.use_dynamic_collate:
            raise NotImplementedError('DynamicCollate is not supported for FunBase')
        elif args.dynamic_flash_attn:
            raise NotImplementedError('DynamicFlashAttn is not supported for FunBase')
    elif args.use_chunkflow or args.use_smart_batching:
        raise NotImplementedError('ChunkFlow and SmartBatching are not supported for FunBase')
    elif args.use_dynamic_collate:
        raise NotImplementedError('DynamicCollate is not supported for FunBase')
    else:
        batch = {}
        if mpu.is_pipeline_first_stage():
            receive_start = time.time()
            batch['input_ids'] = broadcast_anything(None)
            batch['input_ids_audio'] = broadcast_anything(None)
            batch['labels_ids'] = broadcast_anything(None)
            batch['input_ids_codec_text'] = broadcast_anything(None)
            batch['labels_ids_audio'] = broadcast_anything(None)
            batch['loss_mask'] = broadcast_anything(None)
            batch['loss_mask_audio'] = broadcast_anything(None)
            batch['attention_mask'] = broadcast_anything(None)
            batch['position_ids'] = broadcast_anything(None)
            for key in AUDIO_BATCH_KEYS:
                batch[key] = broadcast_anything(None)
            batch['cu_seqlens'] = broadcast_anything(None)
            _codex_log_batch_phase('tp_nonzero_receive_broadcast_batch', time.time() - receive_start, batch)
        elif args.dynamic_flash_attn:
            raise NotImplementedError('DynamicFlashAttn is not supported for FunBase')
    if args.use_chunkflow or args.use_smart_batching:
        raise NotImplementedError('ChunkFlow and SmartBatching are not supported for FunBase')
    elif args.dynamic_flash_attn:
        raise NotImplementedError('DynamicFlashAttn is not supported for FunBase')
    _codex_log_batch_phase('get_batch_on_this_tp_rank_total', time.time() - total_start, batch)
    return batch

def update_sequence_chunks_sp(batch):
    args = get_args()
    if args.sequence_parallel and len(batch) > 0:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        world_size = mpu.get_tensor_model_parallel_world_size()
        if world_size == 1:
            return batch
        sequence_length = 0
        for chunk in batch['sequence_chunks']:
            sequence_length += chunk['length']
        assert sequence_length % world_size == 0, f'sequence_length {sequence_length} should be divisible by tensor parallel size {world_size}'
        chunk_length_sp = sequence_length // world_size
        chunk_begin_sp = chunk_length_sp * tp_rank
        chosen_chunk = None
        for chunk in batch['sequence_chunks']:
            chunk_offset = chunk['offset']
            if chunk_begin_sp >= chunk_offset and chunk_begin_sp + chunk_length_sp <= chunk_offset + chunk['length']:
                chosen_chunk = chunk
                chosen_chunk['begin'] = chunk['begin'] + chunk_begin_sp
                chosen_chunk['length'] = chunk_length_sp
                break
        assert chosen_chunk is not None, f'chosen_chunk is not found'
        batch['sequence_chunks'] = [chosen_chunk]
    return batch

def get_batch_on_this_cp_rank(batch, skip_keys=['packed_seq_params', 'kwargs']):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """
    args = get_args()
    cp_split_size = 1 if args.context_parallel_strategy in ['ulysses', 'kvgather'] else 2
    cp_size = args.context_parallel_size
    if len(batch) > 0 and 'cu_seqlens' in batch and (batch['cu_seqlens'] is not None):
        cu_seqlens = batch['cu_seqlens']
        if isinstance(cu_seqlens, torch.Tensor):
            cu_seqlens = cu_seqlens.to(device='cuda', dtype=torch.int32)
        else:
            cu_seqlens = torch.tensor(cu_seqlens, device='cuda', dtype=torch.int32)
        max_seqlen_val = max_seqlen_from_cu_seqlens(cu_seqlens)
        batch['packed_seq_params'] = {'cu_seqlens_q': cu_seqlens, 'cu_seqlens_kv': cu_seqlens, 'max_seqlen_q': max_seqlen_val, 'max_seqlen_kv': max_seqlen_val, 'qkv_format': 'thd'}
    elif cp_size > 1 and len(batch) > 0:
        if args.context_parallel_strategy in ('kvgather', 'ulysses'):
            cu_seqlens = torch.arange(0, batch['input_ids'].shape[0] + 1, device='cuda', dtype=torch.int32) * batch['input_ids'].shape[1]
            max_seqlen = torch.tensor([batch['input_ids'].shape[1]], device='cuda', dtype=torch.int32)
            batch['packed_seq_params'] = {'cu_seqlens_q': cu_seqlens, 'cu_seqlens_kv': cu_seqlens, 'max_seqlen_q': max_seqlen, 'max_seqlen_kv': max_seqlen, 'qkv_format': 'thd'}
        cp_rank = mpu.get_context_parallel_rank()
        speech_token_nums = None
        for key, val in batch.items():
            if key == 'attention_mask' and args.context_parallel_strategy in ['ulysses', 'kvgather']:
                continue
            if key in skip_keys:
                continue
            if val is not None:
                seq_dim = 1 if key != 'attention_mask' else 1
                val = val.view(*val.shape[0:seq_dim], cp_split_size * cp_size, val.shape[seq_dim] // (cp_split_size * cp_size), *val.shape[seq_dim + 1:])
                if key == 'input_ids' and 'input_ids_audio' not in batch or key == 'input_ids_audio':
                    speech_token_nums = torch.sum(val == -50, axis=2)
                if cp_split_size == 1:
                    index = torch.tensor([cp_rank], device='cpu', pin_memory=True).cuda(non_blocking=True)
                else:
                    index = torch.tensor([cp_rank, cp_split_size * cp_size - cp_rank - 1], device='cpu', pin_memory=True).cuda(non_blocking=True)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[seq_dim + 2:])
                batch[key] = val
        if speech_token_nums is not None:
            batch['speech_token_nums_per_sequence_chunk'] = speech_token_nums.tolist()
        assert len(batch['sequence_chunks']) == 1, 'sequence_chunks should be a list with one element'
        chunk = batch['sequence_chunks'][0]
        chunk_length = chunk['length']
        assert chunk_length % (cp_split_size * cp_size) == 0, f'input chunk length {chunk_length} should be divisible by (cp_split_size {cp_split_size} * cp_size {cp_size}) = {cp_split_size * cp_size}'
        chunk_length_cp = chunk_length // (cp_split_size * cp_size)
        chunks_cp = []
        chunks_cp.append({'index': cp_rank, 'begin': chunk_length_cp * cp_rank, 'length': chunk_length_cp, 'offset': 0})
        if cp_split_size > 1:
            chunks_cp.append({'index': cp_split_size * cp_size - cp_rank - 1, 'begin': chunk_length_cp * (cp_split_size * cp_size - cp_rank - 1), 'length': chunk_length_cp, 'offset': chunk_length_cp})
        batch['sequence_chunks'] = chunks_cp
    return batch

def get_batch(data_iterator):
    """Generate a batch."""
    tp_start = time.time()
    batch = get_batch_on_this_tp_rank(data_iterator)
    _codex_log_batch_phase('get_batch_after_tp', time.time() - tp_start, batch)
    cp_start = time.time()
    batch = get_batch_on_this_cp_rank(batch, skip_keys=['packed_seq_params', 'kwargs', 'segment_fbank', 'segment_fbank_lengths', 'speech', 'speech_lengths', 'speech_info', 'feature_beg', 'feature_type', 'fake_token_len', 'items', 'speech_token_id_offset', 'sequence_chunks', 'target_audio', 'target_audio_lengths', 'target_spk_embedding'])
    _codex_log_batch_phase('get_batch_after_cp', time.time() - cp_start, batch)
    sp_start = time.time()
    batch = update_sequence_chunks_sp(batch)
    _codex_log_batch_phase('get_batch_after_sp', time.time() - sp_start, batch)
    return batch

def loss_func(loss_mask, loss_mask_audio, output_tensor, *extra_outputs):
    """Loss function for FunBase model."""
    args = get_args()
    global _loss_func_debug_logged
    cosyvoice_codex_losses = None
    cosyvoice_codex_loss_mask = None
    output_tuple_len = len(output_tensor) if isinstance(output_tensor, tuple) else 0
    if isinstance(output_tensor, tuple):
        is_dual_channel = True
        if len(output_tensor) == 2:
            text_losses, audio_losses = output_tensor
        elif len(output_tensor) == 3:
            text_losses, audio_losses, cosyvoice_codex_losses = output_tensor
        elif len(output_tensor) == 4:
            text_losses, audio_losses, cosyvoice_codex_losses, cosyvoice_codex_loss_mask = output_tensor
        elif len(output_tensor) == 5:
            text_losses, audio_losses, cosyvoice_codex_losses, cosyvoice_codex_loss_mask, cosyvoice_fm_loss = output_tensor
        else:
            raise ValueError(f'Unsupported output_tensor tuple length: {len(output_tensor)}')
    else:
        is_dual_channel = False
        text_losses = output_tensor
        audio_losses = None
    if not (isinstance(output_tensor, tuple) and len(output_tensor) == 5):
        cosyvoice_fm_loss = None
    text_losses = text_losses.float()
    audio_losses = audio_losses.float() if is_dual_channel else None
    has_cosyvoice_codex_loss = cosyvoice_codex_losses is not None
    cosyvoice_codex_losses = cosyvoice_codex_losses.float() if has_cosyvoice_codex_loss else None
    cosyvoice_codex_loss_mask = cosyvoice_codex_loss_mask.view(-1).float() if cosyvoice_codex_loss_mask is not None else None
    audio_loss_weight = float(getattr(args, 'audio_loss_weight', 1.0))
    text_loss_weight = float(getattr(args, 'text_loss_weight', 1.0))
    if loss_mask is None:
        assert len(extra_outputs) > 0, 'loss_mask is None, but extra_outputs is empty'
        loss_mask = extra_outputs[0]
        loss_mask_audio = extra_outputs[1]
    loss_mask = loss_mask.view(-1).float()
    loss_mask_audio = loss_mask_audio.view(-1).float() if is_dual_channel else None
    if has_cosyvoice_codex_loss and cosyvoice_codex_loss_mask is None:
        cosyvoice_codex_loss_mask = loss_mask_audio
    total_tokens = loss_mask.sum()
    if args.context_parallel_size > 1:
        if is_dual_channel:
            text_losses_tensor = torch.cat([torch.sum(text_losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
            torch.distributed.all_reduce(text_losses_tensor, group=mpu.get_context_parallel_group())
            text_losses = text_losses_tensor[0]
            num_tokens = text_losses_tensor[1]
            audio_losses_tensor = torch.cat([torch.sum(audio_losses.view(-1) * loss_mask_audio).view(1), loss_mask_audio.sum().view(1)])
            torch.distributed.all_reduce(audio_losses_tensor, group=mpu.get_context_parallel_group())
            audio_losses = audio_losses_tensor[0]
            num_tokens_audio = audio_losses_tensor[1]
            if audio_loss_weight == 1.0 and text_loss_weight == 1.0:
                text_loss = text_losses / (num_tokens + num_tokens_audio)
                audio_loss = audio_losses / (num_tokens + num_tokens_audio)
                loss = text_loss + audio_loss
            else:
                text_loss = text_losses / torch.clamp(num_tokens, min=1.0)
                audio_loss = audio_losses / torch.clamp(num_tokens_audio, min=1.0)
                loss = text_loss_weight * text_loss if text_loss_weight != 0.0 else text_loss.detach() * 0.0
                if audio_loss_weight != 0.0:
                    loss = loss + audio_loss_weight * audio_loss
            if has_cosyvoice_codex_loss:
                cosyvoice_codex_losses_tensor = torch.cat([torch.sum(cosyvoice_codex_losses.view(-1) * cosyvoice_codex_loss_mask).view(1), cosyvoice_codex_loss_mask.sum().view(1)])
                torch.distributed.all_reduce(cosyvoice_codex_losses_tensor, group=mpu.get_context_parallel_group())
                num_tokens_cosyvoice = torch.clamp(cosyvoice_codex_losses_tensor[1], min=1.0)
                cosyvoice_codex_loss = cosyvoice_codex_losses_tensor[0] / num_tokens_cosyvoice
                cosyvoice_codex_loss_weight = getattr(args, 'cosyvoice_codex_loss_weight', 1.0)
                if cosyvoice_codex_loss_weight != 0.0:
                    loss = loss + cosyvoice_codex_loss_weight * cosyvoice_codex_loss
        else:
            loss = torch.cat([torch.sum(text_losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
            loss = loss[0] / loss[1]
    elif is_dual_channel:
        text_loss_sum = torch.sum(text_losses.view(-1) * loss_mask)
        audio_loss_sum = torch.sum(audio_losses.view(-1) * loss_mask_audio)
        if audio_loss_weight == 1.0 and text_loss_weight == 1.0:
            text_loss = text_loss_sum / (loss_mask.sum() + loss_mask_audio.sum())
            audio_loss = audio_loss_sum / (loss_mask.sum() + loss_mask_audio.sum())
            loss = text_loss + audio_loss
        else:
            text_loss = text_loss_sum / torch.clamp(loss_mask.sum(), min=1.0)
            audio_loss = audio_loss_sum / torch.clamp(loss_mask_audio.sum(), min=1.0)
            loss = text_loss_weight * text_loss if text_loss_weight != 0.0 else text_loss.detach() * 0.0
            if audio_loss_weight != 0.0:
                loss = loss + audio_loss_weight * audio_loss
        if has_cosyvoice_codex_loss:
            num_tokens_cosyvoice = torch.clamp(cosyvoice_codex_loss_mask.sum(), min=1.0)
            cosyvoice_codex_loss = torch.sum(cosyvoice_codex_losses.view(-1) * cosyvoice_codex_loss_mask) / num_tokens_cosyvoice
            cosyvoice_codex_loss_weight = getattr(args, 'cosyvoice_codex_loss_weight', 1.0)
            if cosyvoice_codex_loss_weight != 0.0:
                loss = loss + cosyvoice_codex_loss_weight * cosyvoice_codex_loss
    else:
        loss = torch.sum(text_losses.view(-1) * loss_mask) / loss_mask.sum()
    if cosyvoice_fm_loss is not None:
        loss = loss + float(getattr(args, 'cosyvoice_fm_loss_weight', 1.0)) * cosyvoice_fm_loss.float()
    if args.reduce_loss_per_iteration:
        averaged_loss = [loss.clone().detach().view(1)]
        if is_dual_channel:
            averaged_text_loss = [text_loss.clone().detach().view(1)]
            averaged_audio_loss = [audio_loss.clone().detach().view(1)]
            if has_cosyvoice_codex_loss:
                averaged_cosyvoice_codex_loss = [cosyvoice_codex_loss.clone().detach().view(1)]
    else:
        averaged_loss = average_losses_across_data_parallel_group([loss])
        if is_dual_channel:
            averaged_text_loss = average_losses_across_data_parallel_group([text_loss])
            averaged_audio_loss = average_losses_across_data_parallel_group([audio_loss])
            if has_cosyvoice_codex_loss:
                averaged_cosyvoice_codex_loss = average_losses_across_data_parallel_group([cosyvoice_codex_loss])
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), f'Rank {global_rank}: found NaN in local forward loss calculation. Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
    if loss.isnan() or loss == 0 or loss < 1e-06:
        logger.info(f'Rank {torch.distributed.get_rank()}: found NaN in local forward loss calculation.')
    loss_dict = {'lm_loss': averaged_loss[0]}
    if is_dual_channel:
        loss_dict.update({'text_loss': averaged_text_loss[0], 'audio_loss': averaged_audio_loss[0]})
        if has_cosyvoice_codex_loss:
            loss_dict.update({'cosyvoice_codex_loss': averaged_cosyvoice_codex_loss[0]})
        if cosyvoice_fm_loss is not None:
            loss_dict['cosyvoice_fm_loss'] = average_losses_across_data_parallel_group([cosyvoice_fm_loss.detach().float()])[0]
    if not _loss_func_debug_logged:
        _loss_func_debug_logged = True
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            loss_mask_audio_sum = float(loss_mask_audio.sum().detach().cpu()) if loss_mask_audio is not None else None
            cosyvoice_codex_loss_mask_sum = float(cosyvoice_codex_loss_mask.sum().detach().cpu()) if cosyvoice_codex_loss_mask is not None else None
            logger.info('[LOSS DEBUG] rank=%s output_tuple_len=%s is_dual_channel=%s has_cosyvoice_codex_loss=%s loss_mask_sum=%s loss_mask_audio_sum=%s cosyvoice_codex_loss_mask_sum=%s loss=%s text_loss=%s audio_loss=%s cosyvoice_codex_loss=%s finite=(%s,%s,%s,%s) loss_keys=%s', rank, output_tuple_len, is_dual_channel, has_cosyvoice_codex_loss, float(loss_mask.sum().detach().cpu()), loss_mask_audio_sum, cosyvoice_codex_loss_mask_sum, float(loss.detach().float().cpu()), float(text_loss.detach().float().cpu()) if is_dual_channel else None, float(audio_loss.detach().float().cpu()) if is_dual_channel else None, float(cosyvoice_codex_loss.detach().float().cpu()) if has_cosyvoice_codex_loss else None, bool(torch.isfinite(loss.detach()).cpu()), bool(torch.isfinite(text_loss.detach()).cpu()) if is_dual_channel else None, bool(torch.isfinite(audio_loss.detach()).cpu()) if is_dual_channel else None, bool(torch.isfinite(cosyvoice_codex_loss.detach()).cpu()) if has_cosyvoice_codex_loss else None, sorted(loss_dict.keys()))
        except Exception as exc:
            logger.info('[LOSS DEBUG] failed to log loss debug info: %s', exc)
    return (loss * args.context_parallel_size, loss_dict)

def moe_loss_func(loss_mask, loss_mask_audio, output_tensor, *extra_outputs):
    loss, loss_dict = (None, {})
    if mpu.is_pipeline_last_stage():
        assert output_tensor is not None
        loss, loss_dict = loss_func(loss_mask, loss_mask_audio, output_tensor, *extra_outputs)
    args = get_args()
    if args.recompute_granularity is not None:
        load_balancing_loss_data = get_load_balancing_loss()
        if args.num_layers * 2 == len(load_balancing_loss_data):
            raise NotImplementedError
    lbl, local_num_tokens_for_lbl = batched_load_balancing_loss()
    local_num_tokens_for_lbl = local_num_tokens_for_lbl.to(torch.int)
    rzl = batched_router_z_loss()
    clear_load_balancing_loss()
    if args.moe_router_sync_tokens_per_expert_across_dp:
        lbl, sequence_lbl = lbl
    lbl = lbl * args.moe_aux_loss_coeff
    clear_router_z_loss()
    if args.reduce_loss_per_iteration:
        averaged_lbl = [lbl.clone().detach().view(1)]
    else:
        averaged_lbl = average_losses_across_data_parallel_group([lbl], with_context_parallel=True)
    loss_dict['load balancing loss'] = averaged_lbl[0].cuda()
    if args.reduce_loss_per_iteration:
        averaged_rzl = [rzl.clone().detach().view(1)]
    else:
        averaged_rzl = average_losses_across_data_parallel_group([rzl], with_context_parallel=True)
    loss_dict['router z loss'] = averaged_rzl[0].cuda()
    if args.moe_expert_capacity_factor > 0:
        dtr = batched_dropped_token_ratio()
        clear_num_dropped_tokens()
        if args.reduce_loss_per_iteration:
            averaged_dtr = [dtr.clone().detach().view(1)]
        else:
            averaged_dtr = average_losses_across_data_parallel_group([dtr], with_context_parallel=True)
        loss_dict['dropped ratio'] = averaged_dtr[0]
    total_loss = loss + lbl + rzl if loss is not None else lbl + rzl
    if mpu.is_pipeline_last_stage():
        return (total_loss, loss_dict)
    else:
        return (total_loss, local_num_tokens_for_lbl, loss_dict)

def forward_step(data_iterator, model: FunBaseModel):
    """Forward training step for FunBase model.

    Args:
        data_iterator : Input data iterator
        model (FunBaseModel): The FunBase Model
    """
    args = get_args()
    timers = get_timers()
    timers('batch-generator', log_level=2).start()
    global stimer
    batch_start = time.time()
    with stimer(bdata=True):
        batch = get_batch(data_iterator)
    timers('batch-generator').stop()
    _codex_log_slow_batch(batch, time.time() - batch_start)
    with stimer:
        output_tensor = model(**batch)
    loss_mask = batch.get('loss_mask', None)
    loss_mask_audio = batch.get('loss_mask_audio', None)
    if args.num_experts is not None:
        return (output_tensor, partial(moe_loss_func, loss_mask, loss_mask_audio))
    else:
        return (output_tensor, partial(loss_func, loss_mask, loss_mask_audio))

def core_audio_dataset_config_from_args(args):
    """Create audio dataset configuration from arguments."""
    tokenizer = get_tokenizer()
    return FunASRDatasetConfig(random_seed=args.seed, sequence_length=args.seq_length, blend=get_blend_from_list(args.train_data_path), blend_per_split=None, split=args.split, num_dataset_builder_threads=args.num_dataset_builder_threads, path_to_cache=args.data_cache_path, mmap_bin_files=args.mmap_bin_files, tokenizer=tokenizer)

def is_dataset_built_on_rank():
    """Check if dataset should be built on this rank."""
    return mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_first_stage()

def convert_batch_to_megatron(input_data, megatron_padding_id, megatron_tokenizer_codec_offset, cosyvoice_padding_id=-1, cosyvoice_ignore_id=-100):
    convert_start = time.time()
    args = get_args()
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = args.context_parallel_size
    cp_split_size = 1 if args.context_parallel_strategy in ['ulysses', 'kvgather'] else 2
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    pad_divisible_by = cp_split_size * cp_size * tp_size
    token_num = input_data['input_ids'].shape[1]
    seq_length = args.seq_length
    token_num = input_data['input_ids'].shape[1]
    if token_num > seq_length:
        raise ValueError(f'token_num {token_num} > seq_length {seq_length}')
    if not args.fun_no_pad_sample_to_seq_length:
        token_num_padded = seq_length
    else:
        token_num_padded = (token_num + pad_divisible_by - 1) // pad_divisible_by * pad_divisible_by
    input_ids_audio = input_data.get('input_ids_audio', None)
    labels_ids_audio = input_data.get('labels_ids_audio', None)
    if token_num <= token_num_padded:
        PAD_ID = -1
        pad_length = token_num_padded - token_num + 1
        input_data['input_ids'] = torch.cat([input_data['input_ids'], torch.ones(input_data['input_ids'].shape[0], pad_length, dtype=torch.int64, device=input_data['input_ids'].device) * PAD_ID], dim=1)[:, :-1]
        input_data['labels_ids'] = torch.cat([input_data['labels_ids'], torch.ones(input_data['labels_ids'].shape[0], pad_length, dtype=torch.int64, device=input_data['labels_ids'].device) * PAD_ID], dim=1)[:, 1:]
        input_data['attention_mask'] = torch.cat([input_data['attention_mask'], torch.zeros(input_data['attention_mask'].shape[0], pad_length, dtype=torch.int64, device=input_data['attention_mask'].device)], dim=1)[:, :-1]
        if input_ids_audio is not None:
            input_data['input_ids_audio'] = torch.cat([input_ids_audio, torch.ones(input_ids_audio.shape[0], pad_length, dtype=input_ids_audio.dtype, device=input_ids_audio.device) * PAD_ID], dim=1)[:, :-1]
        if labels_ids_audio is not None:
            ignore_pad = torch.ones(labels_ids_audio.shape[0], pad_length, dtype=labels_ids_audio.dtype, device=labels_ids_audio.device) * cosyvoice_ignore_id
            input_data['labels_ids_audio'] = torch.cat([labels_ids_audio, ignore_pad], dim=1)[:, 1:]
    position_ids = torch.arange(input_data['attention_mask'].shape[1], dtype=torch.int64).unsqueeze(0).expand(input_data['attention_mask'].shape[0], -1)
    try:
        input_ids_codec_text = build_codec_text_input_ids(input_data['labels_ids'], input_data.get('labels_ids_audio', None))
    except CodecTextRetokenizeError:
        print(f'[BAD_CODEC_TEXT_SAMPLE] items={_codex_summarize_items(input_data.get('item'), limit=8)} data={_codex_data_summary(input_data)}', flush=True)
        raise
    sequence_chunks = [{'begin': 0, 'length': token_num_padded, 'offset': 0}]
    speech = input_data.get('speech', None)
    speech_lengths = input_data.get('speech_lengths', None)
    speech_batch_size_bucket = 32
    speech_length_bucket = 64
    if not args.fun_no_pad_speech_to_bucket:
        pad_speech_start = time.time()
        speech, speech_lengths = pad_speech_to_bucket(input_data.get('speech', None), input_data.get('speech_lengths', None), speech_batch_size_bucket, speech_length_bucket)
        _codex_log_iterator_phase('convert_batch_pad_speech_to_bucket', time.time() - pad_speech_start, input_data, extra={'token_num': int(token_num), 'token_num_padded': int(token_num_padded), 'speech_batch_size_bucket': speech_batch_size_bucket, 'speech_length_bucket': speech_length_bucket})
    output_data = {'segment_fbank': input_data.get('segment_fbank', None), 'segment_fbank_lengths': input_data.get('segment_fbank_lengths', None), 'speech': speech, 'speech_lengths': speech_lengths, 'speech_info': input_data.get('speech_info', None), 'feature_beg': input_data.get('feature_beg', None), 'feature_type': input_data.get('feature_type', None), 'fake_token_len': input_data.get('fake_token_len', None), 'input_ids': input_data['input_ids'], 'attention_mask': input_data['attention_mask'], 'labels_ids': input_data['labels_ids'], 'input_ids_codec_text': input_ids_codec_text, 'input_ids_audio': input_data.get('input_ids_audio', None), 'labels_ids_audio': input_data.get('labels_ids_audio', None), 'position_ids': position_ids, 'items': input_data['item'], 'speech_token_id_offset': input_data['speech_token_id_offset'], 'loss_mask': (~torch.isin(input_data['labels_ids'], torch.tensor([-100, -1]))).int(), 'loss_mask_audio': (~torch.isin(input_data['labels_ids_audio'], torch.tensor([-100, -1]))).int() if 'labels_ids_audio' in input_data else None, 'sequence_chunks': sequence_chunks}
    if 'cu_seqlens' in input_data:
        output_data['cu_seqlens'] = input_data['cu_seqlens']
    _codex_log_iterator_phase('convert_batch_to_megatron_total', time.time() - convert_start, output_data, extra={'token_num': int(token_num), 'token_num_padded': int(token_num_padded)})
    return output_data

def pad_speech_to_bucket(speech, speech_lengths, batch_size_bucket=16, length_bucket=128):
    if speech is None or speech_lengths is None:
        return (None, None)
    batch_size = speech.shape[0]
    length = speech.shape[1]
    batch_size_padded = (batch_size + batch_size_bucket - 1) // batch_size_bucket * batch_size_bucket
    length_padded = (length + length_bucket - 1) // length_bucket * length_bucket
    speech_padded = torch.zeros(batch_size_padded, length_padded, speech.shape[2], device=speech.device)
    speech_padded[:batch_size, :length] = speech
    speech_lengths_padded = torch.zeros(batch_size_padded, 1, dtype=torch.int32, device=speech_lengths.device)
    speech_lengths_padded[:batch_size] = speech_lengths
    return (speech_padded, speech_lengths_padded)

def get_megatron_tokenizer_info():
    print(' ===== get_megatron_tokenizer_info =====, pid: ', os.getpid())
    megatron_tokenizer = get_tokenizer()
    megatron_padding_id = 151645
    megatron_tokenizer_codec_offset = megatron_tokenizer._tokenizer('<|c0|>')['input_ids'][0]
    del megatron_tokenizer
    return (megatron_padding_id, megatron_tokenizer_codec_offset)

class TrainIterator:

    def __init__(self, dataloader, dataset_conf, train_conf, **kwargs):
        self.kwargs = kwargs
        self.dataloader = dataloader
        self.dataset_conf = dataset_conf
        self.train_conf = train_conf
        self.epoch = 0
        self.total_steps = 0
        self.step_in_epoch = 0
        self.step_in_split = 0
        self.max_epoch = train_conf.get('max_epoch', 1)
        self.data_split_i = 0
        self.dataloader_tr = None
        self.dataloader_tr_iterator = None
        self.dataloader_val = None
        self.data_split_num = self.dataloader.data_split_num
        self.dataloader_build_split_num = int(self.dataset_conf.get('data_split_num', getattr(self.dataloader, 'data_split_num', self.data_split_num)))
        self.data_split_size = self.dataset_conf.get('data_split_size', self.train_conf.get('data_split_size', -1))
        if self.data_split_size > 0:
            self.data_split_num = 999999

    def load_state(self, state_dict):
        logger.info(f'loading dataloader state: {state_dict}')
        self.epoch = state_dict['epoch']
        self.data_split_i = state_dict['data_split_i']
        self.step_in_split = state_dict['step_in_split']
        self.step_in_epoch = state_dict['step_in_epoch']
        self.total_steps = state_dict['total_steps']
        self.data_split_num = state_dict['data_split_num']
        self.data_split_size = self.dataset_conf.get('data_split_size', state_dict.get('data_split_size', -1))
        self.dataloader_build_split_num = int(self.dataset_conf.get('data_split_num', getattr(self.dataloader, 'data_split_num', self.data_split_num)))

    def save_state(self):
        state_dict = {'epoch': self.epoch, 'data_split_i': self.data_split_i, 'step_in_split': self.step_in_split, 'step_in_epoch': self.step_in_epoch, 'total_steps': self.total_steps, 'dataloader_iterator_num_yielded': self.dataloader_tr_iterator._num_yielded if self.dataloader_tr_iterator is not None else -1, 'data_split_num': self.data_split_num, 'dataloader_build_split_num': self.dataloader_build_split_num, 'data_split_size': self.data_split_size, 'max_epoch': self.max_epoch}
        logger.info(f'saving dataloader state: {state_dict}')
        return state_dict

    def __next__(self):
        while True:
            if self.dataloader_tr is None:
                dataloader_split_num = self.dataloader_build_split_num
                if dataloader_split_num <= 0:
                    raise RuntimeError(f'invalid dataloader build data_split_num={dataloader_split_num}')
                build_data_split_i = self.data_split_i % dataloader_split_num
                if build_data_split_i != self.data_split_i:
                    logger.info('wrap data_split_i %s to %s for dataloader split num %s', self.data_split_i, build_data_split_i, dataloader_split_num)
                iterator_extra = {'epoch': self.epoch, 'data_split_i': self.data_split_i, 'build_data_split_i': build_data_split_i, 'step_in_split': self.step_in_split, 'step_in_epoch': self.step_in_epoch, 'total_steps': self.total_steps, 'data_split_num': self.data_split_num, 'dataloader_build_split_num': self.dataloader_build_split_num, 'data_split_size': self.data_split_size, 'dataloader_type': type(self.dataloader).__name__}
                build_start = time.time()
                self.dataloader_tr, _ = self.dataloader.build_iter(self.epoch, data_split_i=build_data_split_i, start_step=self.step_in_split)
                _codex_log_iterator_phase('train_iterator_build_iter', time.time() - build_start, extra={**iterator_extra, 'dataloader_tr_type': type(self.dataloader_tr).__name__})
                iter_start = time.time()
                self.dataloader_tr_iterator = iter(self.dataloader_tr)
                _codex_log_iterator_phase('train_iterator_iter_dataloader', time.time() - iter_start, extra={**iterator_extra, 'dataloader_tr_iterator_type': type(self.dataloader_tr_iterator).__name__})
            try:
                next_extra = {'epoch': self.epoch, 'data_split_i': self.data_split_i, 'step_in_split': self.step_in_split, 'step_in_epoch': self.step_in_epoch, 'total_steps': self.total_steps, 'data_split_num': self.data_split_num, 'dataloader_build_split_num': self.dataloader_build_split_num, 'data_split_size': self.data_split_size}
                next_start = time.time()
                data = next(self.dataloader_tr_iterator)
                _codex_log_iterator_phase('train_iterator_next_dataloader', time.time() - next_start, data, extra=next_extra)
                self.total_steps += 1
                self.step_in_split += 1
                self.step_in_epoch += 1
                convert_start = time.time()
                try:
                    batch = convert_batch_to_megatron(data, **self.kwargs)
                except CodecTextRetokenizeError as exc:
                    print(f'[SKIP_CODEC_TEXT_SAMPLE] reason={exc} epoch={self.epoch} data_split_i={self.data_split_i} step_in_split={self.step_in_split} step_in_epoch={self.step_in_epoch} total_steps={self.total_steps} items={_codex_summarize_items(data.get('item') if isinstance(data, dict) else None, limit=8)}', flush=True)
                    continue
                _codex_log_iterator_phase('train_iterator_convert_batch_to_megatron', time.time() - convert_start, batch, extra={'epoch': self.epoch, 'data_split_i': self.data_split_i, 'step_in_split': self.step_in_split, 'step_in_epoch': self.step_in_epoch, 'total_steps': self.total_steps, 'data_split_num': self.data_split_num, 'data_split_size': self.data_split_size})
                return batch
            except StopIteration:
                _codex_log_iterator_phase('train_iterator_stop_iteration', 0, extra={'epoch': self.epoch, 'data_split_i': self.data_split_i, 'step_in_split': self.step_in_split, 'step_in_epoch': self.step_in_epoch, 'total_steps': self.total_steps, 'data_split_num': self.data_split_num, 'data_split_size': self.data_split_size}, force=True)
                self.dataloader_tr_iterator = None
                self.dataloader_tr = None
                self.data_split_i += 1
                self.step_in_split = 0
                if self.data_split_i >= self.data_split_num:
                    self.epoch += 1
                    self.data_split_i = 0
                    self.step_in_epoch = 0
                if self.epoch >= self.max_epoch:
                    raise StopIteration

    def __iter__(self):
        return self

def train_iterator(dataloader, **kwargs):
    for epoch in range(0, 10):
        for data_split_i in range(0, dataloader.data_split_num):
            dataloader_tr, _ = dataloader.build_iter(epoch, data_split_i=data_split_i, start_step=0)
            for data in dataloader_tr:
                try:
                    data = convert_batch_to_megatron(data, **kwargs)
                except CodecTextRetokenizeError as exc:
                    print(f'[SKIP_CODEC_TEXT_SAMPLE] reason={exc} epoch={epoch} data_split_i={data_split_i} items={_codex_summarize_items(data.get('item') if isinstance(data, dict) else None, limit=8)}', flush=True)
                    continue
                yield data

def val_iterator(dataloader, **kwargs):
    for data in dataloader:
        try:
            data = convert_batch_to_megatron(data, **kwargs)
        except CodecTextRetokenizeError as exc:
            print(f'[SKIP_CODEC_TEXT_SAMPLE] reason={exc} items={_codex_summarize_items(data.get('item') if isinstance(data, dict) else None, limit=8)}', flush=True)
            continue
        yield data

def build_fun_data_iterators(train_val_test_num_samples):
    args = get_args()
    if not is_dataset_built_on_rank():
        logger.info(f'Dataset not built on this rank, skip')
        return (None, None, None)
    config_path = args.fun_dataset_config_path
    print_rank_0(f'Loading fun dataset config from {config_path}')
    kwargs = omegaconf.OmegaConf.load(config_path)
    kwargs = dict(kwargs)
    print_rank_0('Build tokenizer')
    tokenizer = kwargs.get('tokenizer', None)
    if tokenizer is not None:
        tokenizer_class = tables.tokenizer_classes.get(tokenizer)
        tokenizer = tokenizer_class(**kwargs.get('tokenizer_conf', {}))
        kwargs['token_list'] = tokenizer.token_list if hasattr(tokenizer, 'token_list') else None
        kwargs['token_list'] = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else kwargs['token_list']
        vocab_size = len(kwargs['token_list']) if kwargs['token_list'] is not None else -1
        if vocab_size == -1 and hasattr(tokenizer, 'get_vocab_size'):
            vocab_size = tokenizer.get_vocab_size()
    else:
        vocab_size = -1
    kwargs['tokenizer'] = tokenizer
    frontend = kwargs.get('frontend', None)
    kwargs['input_size'] = None
    if frontend is not None:
        frontend_class = tables.frontend_classes.get(frontend)
        frontend = frontend_class(**kwargs.get('frontend_conf', {}))
        kwargs['input_size'] = frontend.output_size() if hasattr(frontend, 'output_size') else None
    kwargs['frontend'] = frontend
    slice_frontend = kwargs.get('slice_frontend', None)
    kwargs['slice_input_size'] = None
    if slice_frontend is not None:
        frontend_class = tables.frontend_classes.get(slice_frontend)
        slice_frontend = frontend_class(**kwargs.get('slice_frontend_conf', {}))
        kwargs['slice_input_size'] = slice_frontend.output_size() if hasattr(slice_frontend, 'output_size') else None
    kwargs['slice_frontend'] = slice_frontend
    print_rank_0('Build dataloader')
    dataloader_class = tables.dataloader_classes.get(kwargs['dataset_conf'].get('dataloader', 'DataloaderMapStyle'))
    dataloader = dataloader_class(rank=mpu.get_data_parallel_rank(), world_size=mpu.get_data_parallel_world_size(), **kwargs)
    _, dataloader_val = dataloader.build_iter(0, data_split_i=0, start_step=0)
    megatron_padding_id, megatron_tokenizer_codec_offset = get_megatron_tokenizer_info()
    fill_token = kwargs['dataset_conf'].get('fill_token')
    pad_id = kwargs['dataset_conf'].get('int_pad_value', -1)
    ignore_id = kwargs['dataset_conf'].get('ignore_id', -100)
    specaug_fill_value = kwargs['dataset_conf'].get('specaug_conf', {}).get('fill_value', -100)
    train_iterator = TrainIterator(dataloader, megatron_padding_id=megatron_padding_id, megatron_tokenizer_codec_offset=megatron_tokenizer_codec_offset, dataset_conf=kwargs['dataset_conf'], train_conf=kwargs['train_conf'])
    return (train_iterator, None, None)
if __name__ == '__main__':
    build_fun_data_iterators.is_distributed = True
    pretrain(build_fun_data_iterators, model_provider, ModelType.encoder_or_decoder, forward_step, args_defaults={'tokenizer_type': 'QWenVLTokenizer'}, use_train_valid_test_dataset_provider_as_data_iterators=True, scale_lr_cond=scale_lr_cond)
