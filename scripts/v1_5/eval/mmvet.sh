#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m llava.eval.model_vqa \
    --model-path /path/to/your_checkpoint_dir \
    --question-file /dataset/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /dataset/eval/mm-vet/images \
    --answers-file /dataset/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /dataset/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /dataset/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --dst /dataset/eval/mm-vet/results/llava-v1.5-7b.json

