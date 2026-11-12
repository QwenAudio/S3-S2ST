# Evaluation and parallel inference

[Back to README](../README.md) · [中文](EVALUATION.zh-CN.md)

Install the environments and model weights as described in the main README. Run the commands below from the repository root.

## FLEURS preparation and evaluation

```bash
envs/inference/bin/python tools/prepare_fleurs.py
bash scripts/infer.sh --mode streaming --preset adaptive --input data/fleurs/test_all.jsonl --output runs/fleurs_stream --gpu 0
bash scripts/evaluate.sh --run fleurs_stream --gpu 0 --full --skip-comet
envs/metrics/bin/python tools/report.py --run runs/fleurs_stream
```

Preparation downloads official metadata and audio, checks SHA256 and builds the paper subset: 945 ZH→EN and 646 EN→ZH examples. For an offline server, populate `cache/downloads/fleurs_hf/` first; pinned filenames and hashes are in `configs/fleurs_subset.json` and `configs/fleurs_archives.json`.

Place evaluation assets under `evaluation/third_party/cer_tool_parallel/models/`, `evaluation/third_party/torch_cache/` and `evaluation/voice_eval/thirdparty/3D-Speaker/pretrained/`. `--skip-comet` leaves COMET and ASR-COMET empty. To enable them, obtain the official XCOMET-XL checkpoint, run `tools/prepare_assets.py`, and omit the flag. Scores are written to `quality/summary.tsv` and averaged equally across directions.

Optional whole-file gain preprocessing and parallel streaming decoding:

```bash
envs/inference/bin/python tools/boost_fleurs.py "<path_to_fleurs_audio>" --output-dir "<path_to_new_audio_directory>"
bash scripts/infer_parallel.sh --output runs/fleurs_parallel --gpus 0,1 --workers-per-gpu 2 --skip-metrics
```

The parallel entry applies gain automatically: RMS target 0.0743324953, peak limit 0.95, boost only. It uses `fixed-0.5`, FM8 and 1 GiB KV cache per engine. The single-worker entry uses the original audio. Gain preprocessing is optional and separate from the published results. Concurrent latency measures throughput conditions; measure single-request latency without competing workers. For separately preprocessed audio, update the input manifest to reference the new files.
