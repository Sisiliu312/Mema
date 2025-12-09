#!/bin/bash
cd /home/data/shika/LLaVA-LayerRouter-ca
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

SPLIT="mmbench_dev_20230712"

/home/data/shika/miniconda3/envs/llava/bin/python -m llava.eval.model_vqa_mmbench \
    --model-path /home/data/shika/LLaVA-LayerRouter-ca/checkpoints/OnlyRouter_one_finetune/llava-v1.5-7b \
    --question-file /home/data/shika/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /home/data/shika/LLaVA/playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

/home/data/shika/miniconda3/envs/llava/bin/python scripts/convert_mmbench_for_submission.py \
    --annotation-file /home/data/shika/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /home/data/shika/LLaVA/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir /home/data/shika/LLaVA/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-13b
