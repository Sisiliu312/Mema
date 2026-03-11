#!/bin/bash
# 对某一 run 下所有带 mm_projector.bin 的 checkpoint 做评估 MME。
# 使用方式示例：
#   CHECKPOINT_ROOT=/home/ly/checkpoints RUN=llava-v1.5-scvm-answerloss-40k ./scripts/v1_5/eval/eval_checkpoints_mme.sh
#
# 目录结构假设为：
#   $CHECKPOINT_ROOT/llava-v1.5-scvm-answerloss-40k/
#     ├─ llava-v1.5-7b/                 # 完整 base 模型（已 merge 好，全量权重）
#     └─ llava-v1.5-7b-pretrain/
#          ├─ mm_projector.bin          # 最新 mm_projector
#          ├─ checkpoint-300/mm_projector.bin
#          ├─ checkpoint-600/mm_projector.bin
#          ├─ checkpoint-900/mm_projector.bin
#          └─ checkpoint-1200/mm_projector.bin
#
# 本脚本会遍历 llava-v1.5-7b-pretrain/checkpoint-*，对每个 step：
#   1) 先 merge：base + checkpoint-*/mm_projector.bin -> merged_ckpts/ckpt-{step}
#   2) 再对 merged 模型跑 MME 评测（与 finetune_task.sh 的 merge 用法一致）

set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$(pwd)"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/checkpoints}"
RUN="${RUN:-llava-v1.5-scvm-answerloss-40k}"

BASE="$CHECKPOINT_ROOT/$RUN"
BASE_FULL="$BASE/llava-v1.5-7b"
PRETRAIN_DIR="$BASE/llava-v1.5-7b-pretrain"
MERGED_DIR="$BASE/merged_ckpts"

MME_ROOT="/dataset/eval/MME"
MME_QUESTION="$MME_ROOT/llava_mme.jsonl"
MME_IMAGE="$MME_ROOT/MME_Benchmark_release_version/MME_Benchmark"

if [ ! -d "$PRETRAIN_DIR" ]; then
  echo "Pretrain directory not found: $PRETRAIN_DIR"
  exit 1
fi

if [ ! -d "$BASE_FULL" ]; then
  echo "Base full model not found: $BASE_FULL"
  exit 1
fi

echo "CHECKPOINT_ROOT: $CHECKPOINT_ROOT"
echo "RUN           : $RUN"
echo "BASE_FULL     : $BASE_FULL"
echo "PRETRAIN_DIR  : $PRETRAIN_DIR"
echo "MERGED_DIR    : $MERGED_DIR"
echo ""

# 收集 checkpoint 步数并排序
CHECKPOINTS=()
for d in "$PRETRAIN_DIR"/checkpoint-*; do
  [ -d "$d" ] || continue
  step="${d##*-}"
  [[ "$step" =~ ^[0-9]+$ ]] && CHECKPOINTS+=( "$step" )
done

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
  echo "No checkpoint-* dirs in $PRETRAIN_DIR"
  exit 1
fi

CHECKPOINTS=( $(printf '%s\n' "${CHECKPOINTS[@]}" | sort -n) )
echo "Found ${#CHECKPOINTS[@]} checkpoints: ${CHECKPOINTS[*]}"
echo ""

for step in "${CHECKPOINTS[@]}"; do
  ckpt_dir="$PRETRAIN_DIR/checkpoint-$step"
  if [ ! -f "$ckpt_dir/mm_projector.bin" ]; then
    echo "Skip $step: no mm_projector.bin in $ckpt_dir"
    continue
  fi

  exp_name="llava-v1.5-7b-step${step}"
  merge_out="$MERGED_DIR/ckpt-$step"
  ans_file="$MME_ROOT/answers/${exp_name}.jsonl"

  echo "========== Step $step =========="
  echo "ckpt_dir  : $ckpt_dir"
  echo "merge_out : $merge_out"
  echo "exp_name  : $exp_name"

  # 1) 先 merge：base + checkpoint 的 mm_projector.bin -> merged_ckpts/ckpt-{step}
  if [ ! -f "$merge_out/config.json" ]; then
    echo "Merging checkpoint-$step with base..."
    mkdir -p "$merge_out"
    python scripts/merge_lora_weights.py \
      --model-path "$ckpt_dir" \
      --model-base "$BASE_FULL" \
      --save-model-path "$merge_out"
  else
    echo "Merged model exists, skip merge: $merge_out"
  fi

  mkdir -p "$MME_ROOT/answers"

  # 2) 对 merged 模型跑 MME 评测（不再传 model-base）
  #    若上一次推理中断导致 answers 文件存在但为空，则仍然重新推理
  if [ ! -s "$ans_file" ]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m llava.eval.model_vqa_loader \
      --model-path "$merge_out" \
      --question-file "$MME_QUESTION" \
      --image-folder "$MME_IMAGE" \
      --answers-file "$ans_file" \
      --temperature 0 \
      --conv-mode vicuna_v1
  else
    echo "Answers exist, skip inference: $ans_file"
  fi

  # 将答案转为 MME 官方格式并计算分数
  cd "$MME_ROOT"
  python convert_answer_to_mme.py --experiment "$exp_name"
  python eval_tool/calculation.py --results_dir "eval_tool/answers/$exp_name"
  cd - >/dev/null 2>&1

  echo ""
done

echo "All checkpoints evaluated on MME."

