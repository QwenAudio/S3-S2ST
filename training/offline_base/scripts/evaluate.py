"""Strict offline infer -> codec/FM/WAV -> audited TN and quality metrics."""
_S3_ROOT = next((p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'env.sh').is_file() and (p / 'tools').is_dir()))
import argparse, hashlib, json, os, shutil, subprocess, sys, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'vendor/streaming_eval'
BASE = Path('/path/to/Qwen-public-0803-hejia')
MP = '/path/to/python'

def load_oss_env():
    raw = subprocess.check_output(['bash', '-c', 'source "$1"; env -0', 'bash', str(ROOT / 'scripts/oss_env.sh')])
    for entry in raw.split(b'\x00'):
        if entry.startswith(b'OSS_'):
            k, v = entry.decode().split('=', 1)
            os.environ[k] = v

def main():
    load_oss_env()
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--language', choices=['zh', 'en'], required=True)
    p.add_argument('--phase', choices=['all', 'postprocess', 'decode'], default='all')
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()
    out = Path(a.output).resolve()
    infer = out / 'inference'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE').exists():
        raise RuntimeError('Refuse completed output')
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)

    def run(name, args, env=None):
        e = os.environ.copy()
        e.update(env or {})
        with (logs / (name + '.log')).open('a') as f:
            subprocess.run(list(map(str, args)), stdout=f, stderr=subprocess.STDOUT, check=True, env=e)
    env = {'PYTHONPATH': f'{ROOT}/vendor/python312:{ROOT}/scripts:{CODE}:{BASE}/deps/python:{ROOT}/vendor/funasr_snapshot', 'OMP_NUM_THREADS': '4'}
    try:
        if a.phase == 'all' and (not (infer / 'INFERENCE_COMPLETE').exists()):
            run('inference', ['bash', ROOT / 'scripts/run_inference.sh', '--model', a.model, '--input', a.input, '--output', infer])
        if not (infer / 'INFERENCE_COMPLETE').is_file():
            raise RuntimeError('Inference is not complete')
        manifest = json.loads((infer / 'RUN_MANIFEST.json').read_text())
        assert manifest['input_sha256'] == hashlib.sha256(Path(a.input).read_bytes()).hexdigest(), 'Input changed since inference'
        assert manifest['model'] == str(Path(a.model).resolve()), 'Model differs from inference'
        rows = [json.loads(x) for x in (infer / 'raw.jsonl').read_text().splitlines()]
        expected = len([x for x in Path(a.input).read_text().splitlines() if x.strip()])
        assert len(rows) == expected and len({x['index'] for x in rows}) == expected
        rr = out / '1best_recog'
        rr.mkdir(exist_ok=True)
        if not (rr / 'codec').exists():
            (rr / 'codec').symlink_to(infer / 'codec', target_is_directory=True)
        run('collect', ['/usr/local/bin/python', CODE / 'collect_results.py', '--inputs', infer / 'raw.jsonl', '--output-dir', rr, '--expected', expected, '--task', 's2st'], env)
        wav = rr / 's2st_wav'
        m = out / 'metrics'
        m.mkdir(exist_ok=True)
        run('decode', ['/usr/local/bin/python', CODE / 'decode_streaming.py', '--codec-dir', rr / 'codec', '--output-dir', wav, '--fm-config-path', CODE / 'fm_config.yaml', '--fm-ckpt-path', '__S3_ROOT__/models/flow_matching/epoch_2_step_780000.pt'.replace('__S3_ROOT__', str(_S3_ROOT)), '--voc-ckpt-path', '__S3_ROOT__/models/vocoder'.replace('__S3_ROOT__', str(_S3_ROOT)), '--jsonl', a.input, '--streaming-metadata-jsonl', infer / 'raw.jsonl', '--funasr-root', ROOT / 'vendor/funasr_snapshot', '--use-oss-cache', '--chunk-size', 15, '--streaming-feed-codec-tokens', 15, '--device', f'cuda:{a.gpu}', '--summary-path', wav / 'decode_summary.json'], env)
        import numpy as np
        import soundfile as sf
        paths = list((wav / 'wav').glob('*.wav'))
        assert len(paths) == expected
        for path in paths:
            x, sr = sf.read(path)
            assert sr == 24000 and x.size and np.isfinite(x).all() and (np.max(np.abs(x)) > 0), path
        summary = json.loads((wav / 'decode_summary.json').read_text())
        assert not summary.get('failed') and (not summary.get('silence_fallback', 0)), summary
        (out / 'DECODE_COMPLETE').write_text(str(expected) + '\n')
        if a.phase == 'decode':
            return

        def metric(name, args):
            run(name, args, {'PYTHONPATH': '', 'CUDA_VISIBLE_DEVICES': str(a.gpu), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        metric('prepare', ['/usr/local/bin/python', CODE / 'metrics/prepare_eval.py', '--jsonl', a.input, '--recog-dir', rr, '--wav-dir', wav / 'wav', '--output-dir', m])
        prepared = json.loads((m / 'prepare_summary.json').read_text())
        assert prepared['voice_pairs'] == expected and prepared['matched_text'] == expected, prepared
        metric('text_bleu', [MP, CODE / 'metrics/score_bleu.py', '--ref', rr / 'label', '--hyp', rr / 'text', '--language', a.language, '--output', m / 'text_bleu.json'])
        metric('text_comet', [MP, CODE / 'metrics/eval_comet.py', m / 'comet_input.jsonl', m / 'comet.json'])
        run('asr', ['bash', CODE / 'third_party/cer_tool_parallel/cal_wer.sh', m / 'wer_ref.txt', wav, a.language, 1], {'PYTHONPATH': '', 'WER_PYTHON': MP, 'WER_GPU_OFFSET': str(a.gpu)})
        merged = max((wav / 'tmp').glob('*/results/merge.out'), key=lambda x: x.stat().st_mtime)
        shutil.copyfile(merged, m / 'asr_details_raw.tsv')
        refkeys = [x.split()[0] for x in (m / 'wer_ref.txt').read_text().splitlines() if x.strip()]
        hypkeys = [Path(x.split('\t')[0]).stem for x in (m / 'asr_details_raw.tsv').read_text().splitlines() if x.strip()]
        assert len(hypkeys) == len(refkeys) == expected and len(set(hypkeys)) == expected and (set(hypkeys) == set(refkeys))
        metric('tn_wer', [MP, CODE / 'metrics/tn_wer.py', '--input', m / 'asr_details_raw.tsv', '--output', m / 'asr_details.tsv', '--report', wav / 'wav_res_ref_text.wer', '--audit', m / 'wer_tn_audit.jsonl', '--lang', a.language])
        metric('asr_bleu', [MP, CODE / 'metrics/score_bleu.py', '--ref', rr / 'label', '--asr-tsv', m / 'asr_details_raw.tsv', '--language', a.language, '--output', m / 'asr_bleu.json'])
        metric('asr_comet_prepare', ['/usr/local/bin/python', CODE / 'metrics/prepare_asr_comet.py', '--recog-dir', rr, '--asr-tsv', m / 'asr_details_raw.tsv', '--target-language', a.language, '--output', m / 'asr_comet_input.jsonl'])
        metric('asr_comet', [MP, CODE / 'metrics/eval_comet.py', m / 'asr_comet_input.jsonl', m / 'asr_comet.json'])
        metric('utmos', [MP, CODE / 'metrics/eval_utmos.py', '--wav-scp', m / 'hyp_wav.scp', '--output', m / 'utmos.json', '--device', 'cuda:0', '--repo', CODE / 'third_party/SpeechMOS', '--cache-dir', '/path/to/torch'])
        scores = {k: json.loads((m / (k + '.json')).read_text()) for k in ['text_bleu', 'asr_bleu', 'comet', 'asr_comet', 'utmos']}
        for k, v in scores.items():
            assert v['num_samples'] == expected, (k, v)
        wer = json.loads((m / 'wer_corpus.json').read_text())
        assert wer['num_samples'] == expected
        scores['corpus_wer_cer'] = wer
        from generation_policy import summarize as summarize_generation
        scores['generation'] = summarize_generation(rows)
        scores.update(expected=expected, language=a.language, wer_report=(wav / 'wav_res_ref_text.wer').read_text(), model=a.model, input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest())
        (out / 'FAILED').unlink(missing_ok=True)
        (out / 'summary.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2))
        (out / 'COMPLETE').write_text('ok\n')
    except BaseException:
        (out / 'FAILED').write_text(traceback.format_exc())
        raise
if __name__ == '__main__':
    main()
