# License scope and third-party components

Original S3-S2ST code and documentation are licensed under the MIT License in [LICENSE](LICENSE), unless otherwise stated. The MIT License does not replace licenses or copyright notices for third-party code, model weights, or datasets.

Bundled third-party code retains its upstream licenses and copyright notices, including notices in source headers. Preserve each applicable LICENSE, NOTICE, and MODEL_LICENSE when redistributing those components. The Megatron license is retained in `code/megatron/LICENSE`; other included license files remain with their respective components. Modifications to third-party code remain subject to applicable upstream terms. This repository grants no additional rights to upstream assets.

FLEURS: https://huggingface.co/datasets/google/fleurs (CC BY 4.0; cite the dataset).
XCOMET-XL, XLM-RoBERTa-XL, Whisper, SeACo-Paraformer, CAMPPlus, ERes2Net, SpeechMOS/UTMOS, Qwen backbones, flow matching and vocoder weights retain their respective upstream terms. The locally inherited FM/vocoder checkpoints do not have a confirmed standalone redistribution license in the inspected experiment metadata; verify permission before publishing those weight files. Model weights are excluded from Git. Their presence in a local working directory does not establish permission to redistribute them.

No private training corpus, object-store credentials, or Python/conda environments are intentionally included. The code and model checkpoints may be released separately after their terms are confirmed.

XCOMET-XL official model card: https://huggingface.co/Unbabel/XCOMET-XL ; CC-BY-NC-SA-4.0, gated access requires account acceptance. Its weight is not included because the source archive checkpoint was missing. The download helper respects Hugging Face authentication.

Bundled FunASR snapshots retain their MIT license and additional component notices in `vendor/FunASR/NOTICE.md` and `vendor/offline_funasr/NOTICE.md`. The 3D-Speaker backend retains its Apache-2.0 license in `evaluation/voice_eval/thirdparty/3D-Speaker/LICENSE`. File-level licenses remain authoritative for embedded third-party code.
