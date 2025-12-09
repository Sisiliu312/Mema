#!/bin/bash
cd /home/data/shika/LLaVA-LayerRouter-ca
export PYTHONWARNINGS="ignore"
SPLIT="QKNorm_all"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-path /home/data/shika/LLaVA-LayerRouter-ca/checkpoints/$SPLIT/llava-v1.5-7b \
    --question-file /home/data/shika/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /home/data/shika/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file /home/data/shika/LLaVA/playground/data/eval/MME/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /home/data/shika/LLaVA/playground/data/eval/MME/

python convert_answer_to_mme.py --experiment llava-v1.5-13b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-13b
