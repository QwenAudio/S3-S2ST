# Upstream license scope

This is a locally modified FunASR 1.0.29 snapshot. The toolkit's upstream MIT license is retained in `LICENSE`; source headers and embedded third-party terms take precedence for their files. Model weights use separate terms in `MODEL_LICENSE` or their model cards.

- FunASR / FunTextProcessing: https://github.com/modelscope/FunASR (MIT for original toolkit code).
- CosyVoice-derived components: https://github.com/FunAudioLLM/CosyVoice (Apache-2.0), retained in `third_party_licenses/LICENSE-CosyVoice`.
- Snake activation: https://github.com/EdwardDixon/snake (MIT), retained in `third_party_licenses/LICENSE-Snake`.
- alias-free-torch filters: https://github.com/junjun3518/alias-free-torch (Apache-2.0), retained in `third_party_licenses/LICENSE-alias-free-torch`.
- Julius sinc adaptation: https://github.com/adefossez/julius (MIT), retained in `third_party_licenses/LICENSE-julius`.

These license copies supplement existing file-level notices; they do not relicense other components. The text-processing directory acknowledges additional upstream projects in its README; those projects retain their own terms.

Embedded codec, vocoder and diffusion adaptations also retain the MIT licenses of BigVGAN (NVIDIA), EnCodec (Meta), F5-TTS (Yushen Chen), Matcha-TTS (Shivam Mehta), HiFi-GAN (Jungil Kong) and vector-quantize-pytorch (Phil Wang), copied under `third_party_licenses/`. Existing Apache-2.0 and other file-level notices remain in force; the parent toolkit MIT license does not replace them.
