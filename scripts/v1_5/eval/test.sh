#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m llava.eval.model_vqa_loader \
    --model-path /path/to/your_checkpoint_dir \
    --question-file /dataset/eval/test/test.jsonl \
    --image-folder /dataset/eval/test/images \
    --answers-file /dataset/eval/test/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

