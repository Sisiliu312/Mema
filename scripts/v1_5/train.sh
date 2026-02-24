#!/bin/bash
cd /code/LLaVA-DSU-dynamic-average
export PYTHONWARNINGS="ignore"
export PYTHONPATH=/code/LLaVA-DSU-dynamic-average
SPLIT="llava-v1.5-dynamic-average-123"

# deepspeed --master_port 29400 --include localhost:0,1 llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /models/vicuna-7b-v1.5 \
#     --version plain \
#     --data_path /dataset/LLaVA-Pretrain/blip_laion_cc_sbu_22k.json \
#     --image_folder /dataset/LLaVA-Pretrain/images \
#     --vision_tower /models/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir /checkpoints/$SPLIT/llava-v1.5-7b-pretrain \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 200 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --logging_steps 1 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --run_name pretrain-ca \

deepspeed --master_port 29400 --include localhost:0,1 llava/train/train_mem.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /models/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /dataset/LLaVA-Tuning/llava_v1_5_mix33k.json \
    --image_folder /dataset/LLaVA-Tuning \
    --vision_tower /models/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /checkpoints/$SPLIT/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /checkpoints/$SPLIT/llava-v1.5-7b-lora-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 20 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --run_name finetune \
    --report_to wandb 

python ./scripts/merge_lora_weights.py \
    --model-path /checkpoints/$SPLIT/llava-v1.5-7b-pretrain \
    --model-base /models/vicuna-7b-v1.5 \
    --save-model-path /checkpoints/$SPLIT/llava-v1.5-7b-pretrain-full

python ./scripts/merge_lora_weights.py \
    --model-path /checkpoints/$SPLIT/llava-v1.5-7b-lora-finetune \
    --model-base /checkpoints/$SPLIT/llava-v1.5-7b-pretrain-full \
    --save-model-path /checkpoints/$SPLIT/llava-v1.5-7b-finetune