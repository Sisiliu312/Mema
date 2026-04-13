
set -e
echo "[DEBUG] llavabench.sh start"
ROOT="/path/to/your_root_directory"
cd "$ROOT"
echo "[DEBUG] ROOT=$ROOT"
export PYTHONWARNINGS="ignore"

python -m llava.eval.model_vqa \
    --model-path /path/to/your_checkpoint_dir \
    --question-file /dataset/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder /dataset/eval/llava-bench-in-the-wild/images \
    --answers-file /dataset/eval/llava-bench-in-the-wild/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

export OPENAI_API_KEY="sk-t2pxTbIDe86FCMV_X4fW69h_4oWPLnyt0FE52mNIQUfsYere9yD0l3xsZgU"

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