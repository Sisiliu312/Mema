#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m llava.eval.model_vqa_loader \
    --model-path /path/to/your_checkpoint_dir \
    --question-file /dataset/eval/pope/llava_pope_test.jsonl \
    --image-folder /dataset/eval/pope/val2014 \
    --answers-file /dataset/eval/pope/answers/llava-v1.5-7b-scvm.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /dataset/eval/pope/coco \
    --question-file /dataset/eval/pope/llava_pope_test.jsonl \
    --result-file /dataset/eval/pope/answers/llava-v1.5-7b-scvm.jsonl

python scripts/v1_5/eval/print.py 