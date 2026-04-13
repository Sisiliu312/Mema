#!/bin/bash
export PYTHONWARNINGS="ignore"

SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path /path/to/your_checkpoint_dir \
    --question-file /dataset/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file /dataset/eval/mmbench_cn/answers/$SPLIT/llava-v1.5-7b.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /dataset/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /dataset/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir /dataset/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir /dataset/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b
