#!/bin/bash
# 多组实验：更换 align_loss_weight、learning_rate、dsu_reduction_ratio，每组训练 + merge（可选测评）
cd /code/LLaVA-scvm-answerloss
export PYTHONWARNINGS="ignore"
export PYTHONPATH=/code/LLaVA-scvm-answerloss

# 实验列表：每行 "align_loss_weight learning_rate dsu_reduction_ratio"（第三项省略时默认 4）
# 可随意增删改行
EXPERIMENTS=(
    "0.05  1e-4"
    "0.05  2e-5"
    "0.05  5e-5"
)

for exp in "${EXPERIMENTS[@]}"; do
    read -r ALIGN_LOSS_WEIGHT LEARNING_RATE <<< "$exp"
    # 用权重、学率、DSU reduction 生成唯一目录名
    SPLIT="llava-v1.5-7b-align${ALIGN_LOSS_WEIGHT}-lr${LEARNING_RATE}"
    echo "=============================================="
    echo "Running: align_loss_weight=$ALIGN_LOSS_WEIGHT, learning_rate=$LEARNING_RATE, -> $SPLIT"
    echo "=============================================="

    # 可选：每组训完后跑 MME 测评（按需取消注释并改 answers 名）
    export CUDA_VISIBLE_DEVICES=1
    python -m llava.eval.model_vqa_loader \
        --model-path /checkpoints/$SPLIT/llava-v1.5-7b \
        --question-file /dataset/eval/MME/llava_mme.jsonl \
        --image-folder /dataset/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
        --answers-file /dataset/eval/MME/answers/llava-v1.5-7b.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1
    cd /dataset/eval/MME/ && python convert_answer_to_mme.py --experiment "llava-v1.5-7b" && cd eval_tool && python calculation.py --results_dir answers/"llava-v1.5-7b" && cd /code/LLaVA-scvm-answerloss
done

echo "All experiments done."