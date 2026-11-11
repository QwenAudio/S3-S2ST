"""Score exactly the text and waveform emitted by one final-evaluation run."""
_S3_ROOT = next((p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'env.sh').is_file() and (p / 'tools').is_dir()))
from pathlib import Path
import sys
import argparse, json, re, hashlib, subprocess, os, socket, fcntl, wave, time
P = Path(__file__).resolve().parents[1]
E = P / 'evaluation'
ORIG = P / 'models'
a = argparse.ArgumentParser()
a.add_argument('--run', required=True)
a.add_argument('--gpu', required=True, type=int)
a.add_argument('--full', action='store_true')
a.add_argument('--skip-comet', action='store_true', help='Explicit partial evaluation: XCOMET weights unavailable; scores remain null')
args = a.parse_args()
run = (P / 'runs' / args.run).resolve()
run.relative_to(P)
quality = run / 'quality'
quality.mkdir(exist_ok=True)
lock = (quality / 'run.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
gpu_lock = (P / 'provenance' / f'{socket.gethostname()}_gpu{args.gpu}.lock').open('a')
fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
stable = 0
deadline = time.monotonic() + 180
while stable < 4:
    used = int(subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=memory.free', '--format=csv,noheader,nounits']).decode().strip())
    stable = stable + 1 if used >= 49152 else 0
    assert time.monotonic() < deadline, ('GPU did not have 48 GiB free', args.gpu, used)
    time.sleep(2)
assert (run / 'STATUS').read_text().startswith('inference_complete') or (run / 'STATUS').read_text() == 'smoke_complete'
inputs = [json.loads(x) for x in (run / 'input.jsonl').read_text().splitlines()]
outputs = [json.loads(x) for x in (run / 'raw.jsonl').read_text().splitlines()]
assert len(inputs) == len(outputs) and [r['index'] for r in outputs] == list(range(len(inputs)))
groups = {}
pairing = []
for i, (inp, row) in enumerate(zip(inputs, outputs)):
    user = next((x for x in inp['messages'] if x['role'] == 'user'))
    content = user.get('content', user.get('text'))
    src_wav = re.search('<\\|startofspeech\\|>!?(.*?)<\\|endofspeech\\|>', content).group(1)
    direction = inp.get('direction') or ('zh2en' if 'English' in content else 'en2zh')
    dataset = inp.get('_eval_dataset', inp.get('dataset', 'smoke')).replace('fleurs_official_validation', 'dev')
    name = f'{dataset}_{direction}'
    assert re.fullmatch('[A-Za-z0-9_]+', name)
    src = inp.get('transcript', inp.get('source', ''))
    ref = next((x.get('content', x.get('text')) for x in inp['messages'] if x['role'] == 'assistant'))
    if not row.get('error'):
        assert row['source'] == src and row['reference'] == ref
    key = row.get('key') or Path(src_wav).name + '_' + hashlib.sha1(src_wav.encode()).hexdigest()[:16]
    pred = row.get('prediction', '')
    w = row.get('live_pcm', {}).get('wav')
    if w:
        assert Path(w).is_file() and Path(w).stat().st_size > 44
    groups.setdefault(name, []).append((inp, key, src, ref, pred, w))
    pairing.append(dict(index=i, key=key, source=src_wav, output_wave=w, trace=row.get('live_pcm', {}).get('trace_path', str(Path(w).parent / 'trace.json')) if w else None, group=name))
assert all((not item['trace'] or Path(item['trace']).is_file() for item in pairing))
wave_cache = {str(Path(x[5]).resolve()): str(Path(x[5]).resolve()) for items in groups.values() for x in items if x[5]}
groups = {name: [(*x[:5], wave_cache[str(Path(x[5]).resolve())] if x[5] else None) for x in items] for name, items in groups.items()}
data = []
for name, items in groups.items():
    manifest = quality / (name + '.jsonl')
    manifest.write_text(''.join((json.dumps(dict(x[0], _eval_key=x[1]), ensure_ascii=False) + '\n' for x in items)))
    data.append(str(manifest))
    recog = quality / (name + '.out') / '1best_recog'
    wavs = recog / 's2st_wav/wav'
    wavs.mkdir(parents=True, exist_ok=True)
    keyed = {x: [] for x in ['text', 'label', 'source']}
    empty = []
    assert len({x[1] for x in items}) == len(items)
    for _, key, src, ref, pred, w in items:
        for field, value in [('text', pred), ('label', ref), ('source', src)]:
            keyed[field].append(key + '\t' + re.sub('\\s+', ' ', value).strip() + '\n')
        dst = wavs / (key + '.wav')
        if not w:
            empty.append(key)
            if not dst.exists():
                with wave.open(str(dst), 'wb') as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(24000)
                    f.writeframes(bytes(4800))
        elif not dst.exists():
            dst.symlink_to(Path(w).resolve())
        if w:
            assert dst.resolve() == Path(w).resolve()
    for field, lines in keyed.items():
        (recog / field).write_text(''.join(lines))
    (recog / 'empty_audio_keys.json').write_text(json.dumps(empty))
(quality / 'data.list').write_text('\n'.join(data) + '\n')
(quality / 'pairing.json').write_text(json.dumps(pairing, indent=2))
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', HF_HOME=str(P / 'cache/huggingface'), TORCH_HOME=str(P / 'evaluation/third_party/torch_cache'), XDG_CACHE_HOME=str(P / 'cache/xdg'), MODELSCOPE_CACHE=str(P / 'cache/modelscope'), MPLCONFIGDIR=str(P / 'cache/matplotlib'), TMPDIR=str(P / 'tmp'), COMET_MODEL=str(P / 'models/xcomet/checkpoints/model.ckpt'), COMET_NUM_GPUS='1', COMET_BATCH_SIZE='2', WER_NUM_JOBS='1', DATA_LIST=str(quality / 'data.list'), UTMOS_PYTHON_BIN=os.environ.get('UTMOS_PYTHON_BIN', sys.executable), WER_TOOL_DIR=str(P / 'evaluation/third_party/cer_tool_parallel'), VOICE_EVAL_SCRIPT=str(E / 'voice_eval/scripts/run_voice_eval.sh'), METRIC_PYTHON_BIN=sys.executable, COMET_PYTHON_BIN=sys.executable, VOICE_DEVICE='0', RUN_UTMOS='1' if args.full else '0', RUN_SPK_SIM='1' if args.full else '0')
utmos_local_enabled = False
utmos_proof = P / 'provenance/UTMOS_LOCAL_VALIDATED.json'
utmos_cache = Path('__S3_ROOT__/cache/utmos_local/READY.json'.replace('__S3_ROOT__', str(_S3_ROOT)))
if utmos_proof.exists() and utmos_cache.exists():
    verified_utmos = json.loads(utmos_proof.read_text())
    local_utmos = json.loads(utmos_cache.read_text())
    utmos_wrapper = P / 'bin/utmos_python_local.sh'
    utmos_local_enabled = verified_utmos['status'] == 'passed' and verified_utmos['wrapper_sha256'] == hashlib.sha256(utmos_wrapper.read_bytes()).hexdigest() and (verified_utmos['cache_manifest_sha256'] == local_utmos['manifest_sha256'])
    if utmos_local_enabled:
        env['UTMOS_PYTHON_BIN'] = str(utmos_wrapper)
manifest = dict(scoring_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), run=str(run), input_sha256=hashlib.sha256((run / 'input.jsonl').read_bytes()).hexdigest(), raw_sha256=hashlib.sha256((run / 'raw.jsonl').read_bytes()).hexdigest(), host=socket.gethostname(), gpu=args.gpu, full_audio_quality=args.full, scorer_hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (E / 'metrics').glob('*') if f.suffix in ['.py', '.sh']}, asr_tool_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (P / 'evaluation/third_party/cer_tool_parallel').glob('*') if f.suffix in ['.py', '.sh']}, metric_wrapper_sha256='caller-environment', restored_package_manifests={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (P / 'metric_runtime').glob('*_RESTORED.json')}, started=time.time())
proof = P / 'provenance/RESIDENT_COMET_VALIDATED.json'
resident = False
if proof.exists():
    verified = json.loads(proof.read_text())
    resident = verified['status'] == 'passed' and verified['adapter_sha256'] == hashlib.sha256((P / 'bin/eval_comet_resident.py').read_bytes()).hexdigest() and (verified['wrapper_sha256'] == 'caller-environment')
if args.skip_comet:
    resident = False
if resident or args.skip_comet:
    env.update(RUN_COMET='0', RUN_ASR_COMET='0')
manifest['skipped_metrics'] = ['comet', 'asr_comet'] if args.skip_comet else []
manifest['resident_comet'] = resident
manifest['utmos_local_enabled'] = utmos_local_enabled
if utmos_local_enabled:
    manifest['utmos_cache_validation'] = verified_utmos
(quality / 'COMMAND.json').write_text(json.dumps(manifest, indent=2))
(quality / 'STATUS').write_text('running')
try:
    with (quality / 'scoring.log').open('w') as f:
        subprocess.run(['bash', str(E / 'metrics/run_metrics.sh'), 'not_applicable', str(quality)], env=env, cwd=quality, stdout=f, stderr=subprocess.STDOUT, check=True)
    if resident:
        jobs = []
        with (quality / 'scoring.log').open('a') as f:
            for name in groups:
                result = quality / (name + '.out')
                metrics = result / 'metrics'
                language = 'en' if name.endswith('zh2en') else 'zh'
                subprocess.run([sys.executable, str(E / 'metrics/prepare_asr_comet.py'), '--recog-dir', str(result / '1best_recog'), '--asr-tsv', str(metrics / 'asr_details_raw.tsv'), '--target-language', language, '--output', str(metrics / 'asr_comet_input.jsonl')], env=env, cwd=quality, stdout=f, stderr=subprocess.STDOUT, check=True)
                for kind in ['asr_comet', 'comet']:
                    jobs.append(dict(input=str(metrics / (kind + '_input.jsonl')), output=str(metrics / (kind + '.json'))))
            spec = dict(jobs=jobs, report=str(quality / 'resident_comet_report.json'))
            (quality / 'resident_comet_spec.json').write_text(json.dumps(spec, indent=2))
            subprocess.run([sys.executable, str(P / 'bin/eval_comet_resident.py'), '--spec', str(quality / 'resident_comet_spec.json')], env=env, cwd=quality, stdout=f, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(E / 'metrics/summarize_eval.py'), str(quality)], env=env, cwd=quality, stdout=f, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((quality / 'summary.json').read_text())
    assert len(summary) == len(groups)
    import math
    for row in summary:
        for field in ['text_bleu', 'asr_bleu', 'wer_cer'] + (['spk_sim', 'utmos'] if args.full else []):
            assert isinstance(row[field], (int, float)) and math.isfinite(row[field]), ('Missing or invalid metric', field, row)
        if args.skip_comet:
            assert row['comet'] is None and row['asr_comet'] is None
    for name, items in groups.items():
        r = json.loads((quality / (name + '.out') / 'metrics/prepare_summary.json').read_text())
        assert r['matched_text'] == r['comet_rows'] == len(items) and r['missing_source'] == r['missing_generated_wav'] == 0
        metrics = quality / (name + '.out') / 'metrics'
        observed = {Path(line.split('\t', 1)[0]).stem for line in (metrics / 'asr_details_raw.tsv').read_text().splitlines() if line.strip()}
        assert observed == {x[1] for x in items}, 'Incomplete ASR coverage'
        for kind in [] if args.skip_comet else ['comet', 'asr_comet']:
            report = json.loads((metrics / (kind + '.json')).read_text())
            assert report['num_samples'] == len(items)
    (quality / 'STATUS').write_text('partial_missing_xcomet' if args.skip_comet else 'complete')
    print('QUALITY_PARTIAL_MISSING_XCOMET' if args.skip_comet else 'QUALITY_COMPLETE', args.run, flush=True)
except BaseException:
    (quality / 'STATUS').write_text('failed')
    raise
