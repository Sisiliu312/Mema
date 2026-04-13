#!/bin/bash
# Multi-run experiment script: vary align_loss_weight, learning_rate, and
# dsu_reduction_ratio; each run performs training + merge (optional evaluation).
cd /code/LLaVA-scvm-answerloss
export PYTHONWARNINGS="ignore"
export PYTHONPATH=/code/LLaVA-scvm-answerloss

# Experiment list: each row is
# "align_loss_weight learning_rate dsu_reduction_ratio"
# (third field defaults to 4 if omitted).
# Feel free to add/remove/edit rows.
EXPERIMENTS=(
    "0.1  1e-4  4"
)

for exp in "${EXPERIMENTS[@]}"; do
    read -r ALIGN_LOSS_WEIGHT LEARNING_RATE DSU_REDUCTION_RATIO <<< "$exp"
    DSU_REDUCTION_RATIO="${DSU_REDUCTION_RATIO:-4}"
    # Build a unique run directory name from weight/lr/reduction settings.
    SPLIT="llava-v1.5-7b-align${ALIGN_LOSS_WEIGHT}-lr${LEARNING_RATE}-rr${DSU_REDUCTION_RATIO}-check"
    echo "=============================================="
    echo "Running: align_loss_weight=$ALIGN_LOSS_WEIGHT, learning_rate=$LEARNING_RATE, dsu_reduction_ratio=$DSU_REDUCTION_RATIO -> $SPLIT"
    echo "=============================================="

    # Restore all visible GPUs before each run. This avoids inheriting
    # CUDA_VISIBLE_DEVICES=0 from previous eval and exposing only one GPU.
    unset CUDA_VISIBLE_DEVICES
    deepspeed --master_port 29400 --include localhost:0,1 llava/train/train_mem.py \
        --deepspeed ./scripts/zero2.json \
        --model_name_or_path /models/llava-v1.5-7b \
        --version v1 \
        --tune_dsu True \
        --align_loss_weight "$ALIGN_LOSS_WEIGHT" \
        --dsu_reduction_ratio "$DSU_REDUCTION_RATIO" \
        --data_path /dataset/LLaVA-Tuning/llava_v1_5_mix20k.json \
        --image_folder /dataset/LLaVA-Tuning \
        --vision_tower /models/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir /checkpoints/$SPLIT/llava-v1.5-7b-pretrain \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 300 \
        --save_total_limit 5 \
        --learning_rate "$LEARNING_RATE" \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb

    python ./scripts/merge_lora_weights.py \
        --model-path /checkpoints/$SPLIT/llava-v1.5-7b-pretrain \
        --model-base /models/llava-v1.5-7b \
        --save-model-path /checkpoints/$SPLIT/llava-v1.5-7b

    # Optional: run MME evaluation after each run
    # (edit answers filename as needed).
    export CUDA_VISIBLE_DEVICES=0
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