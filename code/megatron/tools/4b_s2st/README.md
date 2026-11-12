# Qwen3.5-4B S2ST 常用命令

```bash
cd /path/to/megatron

TRAIN_DIR=/path/to/output
TRAIN_NAME=offline
TALKER_INIT_CKPT=/path/to/talker_native.pth
```

## 开始训练

从 4B base 与初始化 Talker 权重开始，adapter 与 Talker 联训；global batch 64，约 1 epoch：

```bash
test ! -e "$TRAIN_DIR/latest_checkpointed_iteration.txt" || {
  echo "TRAIN_DIR 已有 checkpoint，请使用下面的断点续训命令"
  exit 1
}
TASK=runs/stage2_$(date +%Y%m%d_%H%M%S)
nohup env \
  TTS_INIT_FROM_BASE=1 \
  COSYVOICE_TTS_INIT_CKPT="$TALKER_INIT_CKPT" \
  EXP_DIR="$TRAIN_DIR" \
  EXP_NAME="$TRAIN_NAME" \
  GAS_STEPS=8 LR=5e-6 TRAIN_ITERS=34929 \
  SAVE_INTERVAL=1000 \
  bash tools/4b_s2st/run_4b_s2st_stage.sh tts \
  >"${TASK}.out" 2>"${TASK}.err" &
echo $! | tee "${TASK}.pid"
```

查看训练日志：

```bash
tail -f "${TASK}.out"
```

## 停止训练

```bash
pkill -TERM -f -- "--save ${TRAIN_DIR}"
```

该命令不会立即新增 checkpoint；续训从 `latest_checkpointed_iteration.txt` 指向的最近一次保存继续。

## 断点续训

```bash
TASK=runs/stage2_resume_$(date +%Y%m%d_%H%M%S)
nohup env \
  RESUME_TRAINING=1 \
  INIT_CKPT_DIR="$TRAIN_DIR" \
  EXP_DIR="$TRAIN_DIR" \
  EXP_NAME="$TRAIN_NAME" \
  GAS_STEPS=8 LR=5e-6 TRAIN_ITERS=34929 \
  SAVE_INTERVAL=1000 \
  bash tools/4b_s2st/run_4b_s2st_stage.sh tts \
  >"${TASK}.out" 2>"${TASK}.err" &
echo $! | tee "${TASK}.pid"
```

续训会恢复模型、iteration、optimizer/scheduler 和该实验自己的 dataloader state。

## 转换最新 checkpoint

推理前先把最新 MCore checkpoint 转成 HF：

```bash
ITER=$(cat "$TRAIN_DIR/latest_checkpointed_iteration.txt")
ITER_NAME=$(printf 'iter_%07d' "$ITER")
HF_DIR="${TRAIN_DIR}/${ITER_NAME}_HF"

bash tools/4b_s2st/convert_s2st_checkpoint_to_hf.sh \
  "${TRAIN_DIR}/${ITER_NAME}" \
  "$HF_DIR"
```

输出目录已存在时，请换一个新的 `HF_DIR`；转换脚本不会覆盖已有目录。

## 单条推理测试

`ONE_JSONL` 是只包含一条样本的 JSONL 文件，格式见 `tools/4b_s2st/TESTING.md`。

```bash
ONE_JSONL=/absolute/path/to/one_sample.jsonl
ONE_OUTPUT=/absolute/path/to/one_sample_output
printf '%s\n' "$ONE_JSONL" > /tmp/s2st_single_input.list
printf '%s\n' "$ONE_OUTPUT" > /tmp/s2st_single_output.list

THINKER_DEVICES='[0,1]' \
TALKER_DEVICES='[2]' \
INFER_NJOBS=1 \
bash tools/4b_s2st/run_4b_s2st_inference.sh \
  "$HF_DIR" \
  /tmp/s2st_single_input.list \
  /tmp/s2st_single_output.list
```

生成的 wav 位于：

```text
$ONE_OUTPUT/1best_recog/s2st_wav/wav/
```

## 批量推理测试

`INPUT_LIST` 每行一个 JSONL 路径，`OUTPUT_LIST` 每行一个对应输出目录，两者行数和顺序必须一致。

```bash
INPUT_LIST=/absolute/path/to/input_jsonl.list
OUTPUT_LIST=/absolute/path/to/output_dir.list

THINKER_DEVICES='[0,1]' \
TALKER_DEVICES='[2]' \
INFER_NJOBS=1 \
bash tools/4b_s2st/run_4b_s2st_inference.sh \
  "$HF_DIR" \
  "$INPUT_LIST" \
  "$OUTPUT_LIST"
```


## 9-pair 完整评测

推荐使用单入口脚本。传入实验目录、`iter_NNNNNNN` 或已转换的 HF 目录均可；脚本会自动转换/复用 HF、双向常驻推理、8 GPU 并行音频解码，并支持断点续跑：

```bash
CKPT=/absolute/path/to/experiment/iter_0005000
EVAL_OUT=/path/to/your_eval_name

nohup bash tools/4b_s2st/eval/eval_s2st_checkpoint.sh \
  "$CKPT" "$EVAL_OUT" \
  >"${EVAL_OUT}.launcher.out" 2>"${EVAL_OUT}.launcher.err" &
```

中间 checkpoint 快速比较（跳过 COMET、说话人和 DNSMOS，保留双向 EOS、BLEU、WER/CER、ASR-BLEU）：

```bash
EVAL_PROFILE=core bash tools/4b_s2st/eval/eval_s2st_checkpoint.sh "$CKPT" "$EVAL_OUT"
```

只看 EOS、跳过音频解码和所有指标：

```bash
EVAL_PROFILE=eos bash tools/4b_s2st/eval/eval_s2st_checkpoint.sh "$CKPT" "$EVAL_OUT"
```

中断后执行同一条命令即可续跑。最终结果位于：

```text
$EVAL_OUT/summary.tsv
$EVAL_OUT/summary.json
$EVAL_OUT/eos_summary.tsv
$EVAL_OUT/eos_summary.json
```
