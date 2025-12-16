#!/bin/bash
cd /home/data/shika/LLaVA-LayerRouter-ca
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0
SPLIT="multihead+pastloss"

/home/data/shika/miniconda3/envs/llava/bin/python -m llava.eval.model_vqa_loader \
    --model-path /home/data/shika/LLaVA-LayerRouter-ca/checkpoints/$SPLIT/llava-v1.5-7b \
    --question-file /home/data/shika/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /home/data/shika/LLaVA/playground/data/eval/textvqa/train_images \
    --answers-file /home/data/shika/LLaVA/playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

/home/data/shika/miniconda3/envs/llava/bin/python -m llava.eval.eval_textvqa \
    --annotation-file /home/data/shika/LLaVA//playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /home/data/shika/LLaVA//playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl