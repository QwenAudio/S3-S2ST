"""Offline full-source S2ST using the audited explicit codec-feedback engine."""
_S3_ROOT = next((p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'env.sh').is_file() and (p / 'tools').is_dir()))
import argparse, json, os, sys, hashlib, traceback, fcntl, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'vendor/streaming_eval'
sys.path.insert(0, str(CODE))
_model = os.environ.get('OMNIDRCA2_MODEL_DIR')
if '--model' in sys.argv:
    _model = sys.argv[sys.argv.index('--model') + 1]
_fusion = None
if _model:
    os.environ['OMNIDRCA2_MODEL_DIR'] = _model
    from profiles.layerfusion.layer_fusion_assets import resolve
    _fusion = resolve(_model, mode='auto', profile='continuous')
    os.environ['OFFLINE_LAYER_FUSION'] = '1' if _fusion['enabled'] else '0'
from port_allocator import install as _install_ports
_install_ports()
import infer_deferred_s2st as audited
from terminal_delivery import install as _install_terminal_delivery
_install_terminal_delivery()
from pcm_runtime import PCMRuntime
import inspect, textwrap
_orig = audited.CodecHistoryReplayEngine._listen_talker
_src = inspect.getsource(_orig)
_needle = '                                self._global_codec_tokens.append(token)'
assert _src.count(_needle) == 1
_src = _src.replace(_needle, _needle + '\n                                self._s3_pcm.on_codec(token)')
_ns = {}
exec(compile(textwrap.dedent(_src), __file__, 'exec'), _orig.__globals__, _ns)
audited.CodecHistoryReplayEngine._listen_talker = _ns['_listen_talker']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--text-gpu', type=int, default=0)
    p.add_argument('--thinker-gpu', type=int, default=1)
    p.add_argument('--talker-gpu', type=int, default=2)
    p.add_argument('--encoder-device', default='cuda:0')
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--max-codec-tokens', type=int, default=1500)
    p.add_argument('--prefetch-audio', type=int, default=4)
    p.add_argument('--warmup-input', default='')
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--single-gpu', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    output_lock = (out / 'inference.lock').open('a')
    fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['OMNIDRCA2_MODEL_DIR'] = a.model
    os.environ['CODEX_AUDIO_BLANK_TOKEN_COUNT'] = '5'
    os.environ['CODEX_ALIGNMENT_MASK_TOKEN_ID'] = '-60'
    os.environ['CODEX_TALKER_MAX_CODEC_TEXT_RATIO'] = '0'
    os.environ.pop('VLLM_STREAM_GATE_PATH', None)
    import numpy as np
    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    import infer_deferred_s2st as audited
    from transformers import AutoTokenizer
    from vllm.v1.engine.async_llm import AsyncEngineArgs
    from forced_sequence_logits import ForcedSequenceLogitsProcessor
    encoder = audited.s2tt.FullRecordEncoder(Path(a.model), a.encoder_device, true_streaming=False)
    encoder.processor._frontend = audited.s2tt.VllmWhisperFrontend(fs=16000, n_mels=128, do_pad_trim=False, filters_path=audited.s2tt.MEL_FILTERS_PATH)

    class OfflineBackend(audited.StreamingS2STBackend):

        def __init__(self):
            self.args = a
            self.model_dir = Path(a.model)
            self.tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
            self.im_end_id = int(self.tokenizer.convert_tokens_to_ids('<|im_end|>'))
            common = dict(model=a.model, trust_remote_code=True, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='mp', enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False, max_model_len=8192, max_num_seqs=1, block_size=16)
            mm = dict(limit_mm_per_prompt={'audio': 1}, enable_mm_embeds=True, skip_mm_profiling=True, enable_hidden_state_caching=False)
            self.text_engine = audited.OmniDRCA2Engine(AsyncEngineArgs(**common, **mm, gpu_memory_utilization=0.23 if a.single_gpu else 0.55), thinker_visible_devices=[a.text_gpu])
            self.s2st_engine = audited.CodecHistoryReplayEngine(AsyncEngineArgs(**common, **mm, gpu_memory_utilization=0.23 if a.single_gpu else 0.55, logits_processors=[ForcedSequenceLogitsProcessor]), AsyncEngineArgs(**common, gpu_memory_utilization=0.18 if a.single_gpu else 0.25, enable_prompt_embeds=True), thinker_visible_devices=[a.thinker_gpu], talker_visible_devices=[a.talker_gpu], codec_embed_path=self.model_dir / 'cosyvoice_codec_embed.pt')
            self.codec_eos_id = 59050
            self.codec_bos_id = 59049
            self.codec_final_bos_id = 59049
            self.codec_pause_id = 59051
            self.codec_boundary_policy = 'unified_eos'
            self.codec_envelope = True
            self.final_eos_only = False
            self.codec_history_replay = False
            self.codec_blank_count = 5
            self.codec_alignment_mask_id = -60
            self.audio_pad_id = int(self.tokenizer.convert_tokens_to_ids('<|audio_pad|>'))
            self.s2st_engine.enable_global_timeline()
            self.s2st_engine.global_codec_final_bos_id = 59049
            self.s2st_engine.global_boundary_policy = 'unified_eos'
            self.start_sample()

        def close(self):
            self._close_consumed_writer_shm_handles()
            self.text_engine.shutdown()
            self.s2st_engine.shutdown()
    a.codec_temperature = 0.8
    a.codec_top_p = 0.8
    a.codec_top_k = 5
    rows = [json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]
    if a.limit:
        rows = rows[:a.limit]
    assert 0 <= a.shard_id < a.num_shards
    selected = list(range(a.shard_id, len(rows), a.num_shards))
    manifest = {'layer_fusion': _fusion, 'mode': 'offline_full_source', 'frontend': 'utterance_max', 'streaming_encoder': False, 'gate': False, 'model': str(Path(a.model).resolve()), 'input_sha256': hashlib.sha256(Path(a.input).read_bytes()).hexdigest(), 'expected': len(selected), 'total_input': len(rows), 'shard_id': a.shard_id, 'num_shards': a.num_shards, 'single_gpu': a.single_gpu, 'sampling': {'thinker': 'greedy', 'temperature': 0.8, 'top_p': 0.8, 'top_k': 5, 'seed': a.seed}, 'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / 'scripts/terminal_delivery.py', ROOT / 'scripts/generation_policy.py', ROOT / 'scripts/port_allocator.py', ROOT / 'scripts/offline_audio.py', ROOT / 'scripts/audio_prefetch.py', ROOT / 'scripts/offline_io.py', ROOT / 'scripts/audio_io_checks.py', *CODE.glob('*.py'), *CODE.glob('vllm_overrides/*.py'), *CODE.glob('profiles/layerfusion/*.py')]}}
    if (out / 'RUN_MANIFEST.json').exists():
        prior = json.loads((out / 'RUN_MANIFEST.json').read_text())
        assert all((prior.get(k) == manifest.get(k) for k in ['model', 'input_sha256', 'shard_id', 'num_shards', 'single_gpu'])), 'Cannot resume different model/input/shard/layout'
        with (out / 'PRIOR_MANIFESTS.jsonl').open('a') as f:
            f.write(json.dumps(prior) + '\n')
    manifest['generation_failure_policy'] = 'no-EOS: record failure and decode bounded truncated audio; include in metrics'
    (out / 'RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    (out / 'input.jsonl').write_text(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in rows)))
    (out / 'codec').mkdir(exist_ok=True)
    backend = None
    from generation_policy import generate_sample, summarize
    completed = [json.loads(x) for x in (out / 'raw.jsonl').read_text().splitlines() if x.strip()] if (out / 'raw.jsonl').exists() else []
    assert [x['index'] for x in completed] == selected[:len(completed)] and len(completed) <= len(selected)
    assert all((Path(x['codec_path']).is_file() and Path(x['codec_control_path']).is_file() for x in completed))
    try:
        init_start = time.perf_counter()
        backend = OfflineBackend()
        pcm = PCMRuntime(out / 'pcm', a.encoder_device, dict(base=0.5, cap=0.5, saturation=1.0, paced=False, fm_config='__S3_ROOT__/configs/fm_config.yaml'.replace('__S3_ROOT__', str(_S3_ROOT))))
        backend.s2st_engine._s3_pcm = pcm
        (out / 'STARTUP_TIMING.json').write_text(json.dumps({'backend_init_seconds': time.perf_counter() - init_start}))
        with (out / 'raw.jsonl').open('a') as f:
            from audio_prefetch import ordered_prefetch
            from offline_audio import load_source
            import offline_io
            offline_io.install()
            done_indices = {r['index'] for r in completed}
            warmups = [json.loads(x) for x in Path(a.warmup_input).read_text().splitlines()][:2] if a.warmup_input else []
            pending = list(range(-len(warmups), 0)) + [i for i in selected if i not in done_indices]

            def read_audio(i):
                started = time.perf_counter()
                obj = rows[i] if i >= 0 else warmups[i + len(warmups)]
                wav_path, task, source, reference = audited.s2tt.load_row(obj)
                user = next((m for m in obj['messages'] if m['role'] == 'user'))
                wav = load_source(wav_path, duration=(user.get('start_time', 0), user.get('end_time')))
                return (wav_path, task, source, reference, wav, time.perf_counter() - started)
            previous_end = time.perf_counter()
            for i, data in ordered_prefetch(pending, read_audio, a.prefetch_audio):
                begin = time.perf_counter()
                io_wait = begin - previous_end
                wav_path, task, source, reference, wav, io_work = data
                pcm.root = out / ('warmup_pcm' if i < 0 else 'pcm')
                pcm.root.mkdir(exist_ok=True)
                pcm.begin(i)
                pcm.prefix(wav, len(wav))
                pcm.fix_speaker()
                embeds = encoder(wav)
                encoded_at = time.perf_counter()
                prompt, segments = audited.s2tt.build_prompt(audited.s2tt.SYSTEM, task, embeds, [], open_for_write=True)
                backend.start_sample()
                backend.set_write_context(final_write=True)
                text = backend.generate_text(prompt, segments)
                text_at = time.perf_counter()
                pcm.text_events.append(dict(source_ms=pcm.source_ms, wall_s=time.monotonic() - pcm.play.start, text=text))
                if not text:
                    raise RuntimeError('Empty Thinker output')
                observed_path = out / 'OBSERVED_FAILURES.json'
                observed = json.loads(observed_path.read_text()).get(str(i)) if observed_path.exists() else None
                if observed is not None:
                    assert observed['status'] == 'empty_audio' and observed['eos_observed']
                    codec = np.empty(0, dtype=np.int64)
                    annotations = []
                    backend._global_codec_full = np.empty(0, dtype=np.int64)
                    generation = dict(status='empty_audio', eos_observed=True, truncated=False, error=observed['evidence'])
                else:
                    codec, annotations, generation = generate_sample(backend, prompt, segments, text)
                if not len(codec) and generation['status'] != 'empty_audio' or np.any(codec < 0) or np.any(codec >= 59049):
                    raise RuntimeError('Invalid acoustic codec')
                codec_at = time.perf_counter()
                key = audited.s2tt.make_key(wav_path, i)
                cd = out / ('warmup_codec' if i < 0 else 'codec')
                cd.mkdir(exist_ok=True)
                cp = cd / f'{key}.npy'
                ctrl = cd / f'{key}.control.npy'
                np.save(cp, codec)
                np.save(ctrl, backend._global_codec_full)
                row = dict(generation_status=generation['status'], eos_observed=generation['eos_observed'], truncated=generation['truncated'], generation_error=generation['error'], index=i, key=key, wav=wav_path, source=source, reference=reference, prediction=text, codec_path=str(cp.resolve()), codec_control_path=str(ctrl.resolve()), codec_len=len(codec), timing_s=dict(audio_io_work=io_work, audio_io_wait=io_wait, encoder=encoded_at - begin, text=text_at - encoded_at, codec=codec_at - text_at, total_active=codec_at - begin), codec_segments=annotations, talker_write_count=1, first_write_ratio=1.0, decisions=[], audio_seconds=len(wav) / 16000)
                row['live_pcm'] = pcm.finish(key, truncated=generation['truncated'], allow_empty=generation['status'] == 'empty_audio')
                row['live_pcm']['clock_note'] = 'offline compute after complete source; not streaming wall onset'
                row['live_pcm']['offline_source_plus_compute_onset_s'] = len(wav) / 16000 + row['live_pcm']['startoffset_wall_s'] if row['live_pcm']['startoffset_wall_s'] is not None else None
                if i < 0:
                    print('WARMUP', i, generation['status'], flush=True)
                    previous_end = time.perf_counter()
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                print(i, key, len(codec), generation['status'], row['timing_s'], flush=True)
                previous_end = time.perf_counter()
        all_rows = [json.loads(x) for x in (out / 'raw.jsonl').read_text().splitlines() if x.strip()]
        assert len(all_rows) == len(selected)
        (out / 'GENERATION_SUMMARY.json').write_text(json.dumps(summarize(all_rows), indent=2))
        (out / 'FAILED').unlink(missing_ok=True)
        (out / 'INFERENCE_COMPLETE').write_text(str(len(selected)) + '\n')
    except BaseException:
        (out / 'FAILED').write_text(traceback.format_exc())
        raise
    finally:
        if backend:
            backend.close()
if __name__ == '__main__':
    main()
