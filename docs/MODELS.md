# Model assets

Two model releases are provided: `streaming` and `offline`. Release names have no step or iteration suffix. Install from Hugging Face with `python tools/download_model.py --mode streaming --repo-id "<hf_streaming_repo>"` (use `offline` for the offline release), or use `--local-dir "<path_to_download>"` for manually downloaded files. The installer verifies checksums and places bundles and shared assets inside the repository:

```text
models/
  streaming/             # HF export, audio_encoder.pt, streaming_gate.pt, layer-fusion weights
  offline/               # HF export and audio_encoder.pt
  speaker/campplus.onnx
  flow_matching/          # shared flow-matching weights
  vocoder/                # shared vocoder weights and configuration
  frontend/              # mel_filters.npz and fm_mel_filters.npz
  training/talker_native.pth
checkpoints/
  init_streaming_talker/  # checkpoint parent with latest_checkpointed_iteration.txt
  init_streaming_e2e/
  init_offline/
```

`--model "<path_to_model_export>"` selects the HF export; renderer assets stay under `models/`. `--init "<path_to_checkpoint_parent>"` selects a training initializer. These paths must resolve inside the repository. Raw Megatron checkpoints are for training, HF exports for inference.

Evaluation weights belong under `evaluation/third_party/cer_tool_parallel/models/`, `evaluation/third_party/torch_cache/` and `evaluation/voice_eval/thirdparty/3D-Speaker/pretrained/`; all are Git-ignored. See [third-party notices](../THIRD_PARTY_NOTICES.md). Optional XCOMET: [official checkpoint](https://huggingface.co/Unbabel/XCOMET-XL).
