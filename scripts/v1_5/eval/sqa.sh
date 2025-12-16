#!/bin/bash
cd /home/data/shika/LLaVA-LayerRouter-ca
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0
SPLIT="multihead+pastloss"

/home/data/shika/miniconda3/envs/llava/bin/python -m llava.eval.model_vqa_science \
    --model-path /home/data/shika/LLaVA-LayerRouter-ca/checkpoints/$SPLIT/llava-v1.5-7b \
    --question-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /home/data/shika/LLaVA/playground/data/eval/scienceqa/test \
    --answers-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

/home/data/shika/miniconda3/envs/llava/bin/python llava/eval/eval_science_qa.py \
    --base-dir /home/data/shika/LLaVA/playground/data/eval/scienceqa \
    --result-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json
