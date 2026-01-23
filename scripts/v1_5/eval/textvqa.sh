#!/bin/bash
cd /code/LLaVA-Text-Dynamic-convert
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-path /checkpoints/llava-v1.5-text-Dyn-convert-T-123/llava-v1.5-7b \
    --question-file /dataset/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /dataset/eval/textvqa/train_images \
    --answers-file /dataset/eval/textvqa/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /dataset/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /dataset/eval/textvqa/answers/llava-v1.5-7b.jsonl