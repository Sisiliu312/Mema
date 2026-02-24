#!/bin/bash
cd /code/LLaVA-DSU-dynamic-multi-cl-finetune
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0


python -m llava.eval.model_vqa_mmmu \
    --model_path /checkpoints/llava-v1.5-dynamic-multi-cl-finetune-loss/llava-v1.5-7b \
    --output_path /dataset/eval/MMMU/mmmu/outputs/llava-v1.5-7b_val.json \
    --data_path /dataset/eval/MMMU/ \
    --config_path /dataset/eval/MMMU/mmmu/configs/llava1.5.yaml \
    --split validation \
    --seed 42

python -m llava.eval.eval_mmmu \
    --output_path /dataset/eval/MMMU/mmmu/outputs/llava-v1.5-7b_val.json \
    --answer_path /dataset/eval/MMMU/mmmu/answer_dict_val.json
