#!/bin/bash
set -e
echo "[DEBUG] llavabench.sh start"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
echo "[DEBUG] ROOT=$ROOT"
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

# python -m llava.eval.model_vqa \
#     --model-path /checkpoints/llava-v1.5-dynamic-multi-cl-finetune-loss/llava-v1.5-7b \
#     --question-file /dataset/eval/llava-bench-in-the-wild/questions.jsonl \
#     --image-folder /dataset/eval/llava-bench-in-the-wild/images \
#     --answers-file /dataset/eval/llava-bench-in-the-wild/answers/llava-v1.5-7b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

mkdir -p /dataset/eval/llava-bench-in-the-wild/reviews

echo "[DEBUG] running eval_gpt_review_bench.py..."
python llava/eval/eval_gpt_review_bench.py \
    --question /dataset/eval/llava-bench-in-the-wild/questions.jsonl \
    --context /dataset/eval/llava-bench-in-the-wild/context.jsonl \
    --rule "$ROOT/llava/eval/table/rule.json" \
    --answer-list \
        /dataset/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        /dataset/eval/llava-bench-in-the-wild/answers/llava-v1.5-7b.jsonl \
    --output \
        /dataset/eval/llava-bench-in-the-wild/reviews/llava-v1.5-7b.jsonl

echo "[DEBUG] eval_gpt_review_bench.py done, running summarize_gpt_review.py..."
python llava/eval/summarize_gpt_review.py -f /dataset/eval/llava-bench-in-the-wild/reviews/llava-v1.5-7b.jsonl
echo "[DEBUG] llavabench.sh done"