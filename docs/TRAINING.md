# Training data and stages

The release includes training code, stage configurations, data-construction tools and the public benchmark workflow. Supply your own licensed training data; the original private corpus is not distributed.

Data must be prepared locally under `data/train`. `data.list` has one JSONL shard path per line. Packed examples preserve multi-turn user speech and assistant text, per-record isolation, token lengths, source span boundaries, READ/WRITE supervision and precomputed FSQ exact codec slices. The exact loader is `training/streaming/{talker,e2e}/runtime/exact_codec_loader.py`; local data tools are under `training/data_tools` and `code/megatron/streaming/data`. Inspect those schemas before producing training data. A plain `{audio,text}` file is not a valid replacement. Precomputed codec frame rate is 12.5 Hz, blank count 5, BOS 59049/EOS 59050, and boundary tolerance 80 ms. Each segment uses one BOS and no extra pause token.

Streaming selected main data source durations: 6516.23h ZH→EN, 6505.68h EN→ZH, balanced by duration. ASR source/target cutoffs .32/.16 and .4375/.13125. Final streaming FLEURS supplement3914 retained examples×10; offline5766×10; 9pair supplement removed in final speech-stage mixtures. Earlier text-stage data differed. Historical corpus statistics describe the paper, not downloadable training data.

Store packed data and its audio/codec dependencies locally. Optional OSS readers are available for datasets that use object storage.

The stage launchers retain BF16, seed1024, sequence8192 and exact module freezing/fusion behavior. Talker stage:5000 updates,8GPU,GBS64,LR5e-6,warmup1000. E2E:checkpoint4000 of planned5000,8GPU,GBS64,LR1e-6,gate1e-5,no warmup. Offline continuation:checkpoint10000 of planned68000,32GPU,GBS64,TalkerLR5e-6,bridge/fusionLR1e-4,warmup100,cosine decay68000,minLR1e-6; source frontend and Thinker frozen. Initialize optimizer/scheduler afresh at stage transition.

For multinode training, synchronize preparation before ranks start. Each node uses the same shared project root and output, with NNODES4/NPROC_PER_NODE8/NODE_RANK0..3/MASTER_ADDR=node0. A single-node 8-GPU adaptation can be run with NNODES1, but it changes hardware/runtime and is not the original32GPU execution. Use `--prepare-only` to generate the configuration before starting distributed training.

Training trajectories depend on the corpus, hardware, data order, kernels and optimizer state.


## Construct streaming trajectories

Supply your own licensed bilingual pairs, semantic segmentation and word-level alignment. No private service credentials or training data are included.

```bash
python training/data_tools/build_interleaved_data.py \
  --direction zh2en --mapping "<path_to_mapping.jsonl>" \
  --text-labels "<path_to_semantic_chunks.jsonl>" \
  --word-alignments "<path_to_word_alignments.jsonl>" \
  --output-dir data/train/zh2en --sqlite cache/zh2en.sqlite
python training/data_tools/pack_streaming_s2tt.py \
  --input-list "<path_to_generated_data.list>" --output-dir data/train/packed \
  --funasr-root vendor/FunASR --tokenizer-model models/offline --cutoff-len 8192
```

Repeat with `--direction en2zh`. `clean_causal_trajectories.py` filters alignment/ASR and adds a non-final block delay. `balance_mix_packed.py` balances directions; `audit_*.py` checks boundaries and pack conservation. Input contracts are in `build_record()`.

This packer constructs streaming S2TT data. S2ST also needs aligned target speech and precomputed exact FSQ slices; follow `exact_codec_loader.py` rather than treating ordinary audio/text or S2TT-only packs as S2ST inputs. Run with `--prepare-only` to validate paths and generate stage configuration before launching training.
