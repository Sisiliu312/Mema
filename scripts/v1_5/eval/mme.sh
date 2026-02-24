#!/bin/bash
cd /code/LLaVA-DSU-dynamic-multi-cl-finetune
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-path /checkpoints/llava-v1.5-dynamic-multi-cl-finetune-loss/llava-v1.5-7b \
    --question-file /dataset/eval/MME/llava_mme.jsonl \
    --image-folder /dataset/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file /dataset/eval/MME/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /dataset/eval/MME/

python convert_answer_to_mme.py --experiment llava-v1.5-13b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-13b 