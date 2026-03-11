#!/bin/bash
cd /code/LLaVA-scvm-answerloss
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-path /checkpoints/llava-v1.5-scvm-answerloss/llava-v1.5-7b \
    --question-file /dataset/eval/MP-DocVQA/question_val.jsonl \
    --image-folder /dataset/eval/MP-DocVQA/images \
    --answers-file /dataset/eval/MP-DocVQA/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


python -m llava.eval.eval_from_predictions \
    --annotation-file /dataset/eval/MP-DocVQA/val.json \
    --result-file /dataset/eval/MP-DocVQA/answers/llava-v1.5-7b.jsonl
