#!/bin/bash
cd /code/LLaVA-Text-Attn
export PYTHONWARNINGS="ignore"
export PYTHONPATH=/code/LLaVA-Text-Attn

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /models/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /dataset/LLaVA-Tuning/llava_v1_5_mix6k.json \
    --image_folder /dataset/LLaVA-Tuning \
    --vision_tower /models/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /checkpoints/llava-v1.5-7b-text-6klora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --diversity_weight 1 \
    --use_ca True \
    --tune_ca True \
    --use_router False \
    --tune_router False 

python ./scripts/merge_lora_weights.py \
    --model-path /checkpoints/llava-v1.5-7b-text-finetune6k-lora \
    --model-base /models/vicuna-7b-v1.5 \
    --save-model-path /checkpoints/llava-v1.5-7b-text-finetune6k
