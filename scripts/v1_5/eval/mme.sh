#!/bin/bash
cd /code/LLaVA-scvm-answerloss
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-path /checkpoints/llava-v1.5-scvm-answerloss/llava-v1.5-7b \
    --question-file /dataset/eval/MME/llava_mme.jsonl \
    --image-folder /dataset/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file /dataset/eval/MME/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /dataset/eval/MME/

python convert_answer_to_mme.py --experiment llava-v1.5-13b

cd eval_tool

# 默认会输出 Perception（10 子任务）和 Cognition（4 子任务）两个子集分数
python calculation.py --results_dir answers/llava-v1.5-13b

# 仅测评 Perception 子集：python calculation.py --results_dir answers/llava-v1.5-13b --eval_type Perception
# 仅测评 Cognition 子集：python calculation.py --results_dir answers/llava-v1.5-13b --eval_type Cognition