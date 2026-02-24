#!/bin/bash
cd /code/LLaVA-DSU-dynamic-multi-cl-finetune
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_v7w \
    --model-path /checkpoints/llava-v1.5-dynamic-multi-cl-finetune-loss/llava-v1.5-7b \
    --image-folder /dataset/eval/v7w/images \
    --question-file /dataset/eval/v7w/_telling_val_mc.jsonl \
    --answers-file /dataset/eval/v7w/answers/llava-v1.5-7b.jsonl \
    --mc_out /dataset/eval/v7w/answers/my_llava_v7w_mc.json \
    --temperature 0 \
    --num_beams 1 \
    --max_new_tokens 8 \
    --force_no_sample

python -m llava.eval.evaluate_v7w \
    --dataset visual7w-telling \
    --dataset-root /dataset/eval/v7w \
    --mode mc \
    --split val \
    --results /dataset/eval/v7w/answers/my_llava_v7w_mc.json \
    --verbose 1 \
    --topk 1

