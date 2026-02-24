#!/bin/bash
cd /code/LLaVA-DSU-dynamic-multi-cl-finetune
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-path /checkpoints/llava-v1.5-dynamic-multi-cl-finetune-loss/llava-v1.5-7b \
    --question-file /dataset/eval/zerobench/zerobench_llava_questions.jsonl \
    --image-folder /dataset/eval/zerobench \
    --answers-file /dataset/eval/zerobench/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_zerobench \
    --gt-json /dataset/eval/zerobench/zerobench_gt.json \
    --result-file /dataset/eval/zerobench/answers/llava-v1.5-7b.jsonl \
    --mode official
