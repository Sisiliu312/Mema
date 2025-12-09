#!/bin/bash
cd /home/data/shika/LLaVA-LayerRouter-ca
export PYTHONWARNINGS="ignore"
export PYTHONPATH=/home/data/shika/LLaVA-LayerRouter-ca
SPLIT="OnlyRouter_one_finetune"

# /home/data/shika/miniconda3/envs/llava/bin/deepspeed --master_port 29501 --include localhost:0,1,2,3 llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /home/data/shika/models/lmsys/vicuna-7b-v1.5 \
#     --version plain \
#     --data_path /home/data/shika/LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_58k.json \
#     --image_folder /home/data/shika/LLaVA/playground/data/LLaVA-Pretrain/images \
#     --vision_tower /home/data/shika/models/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir ./checkpoints/$SPLIT/llava-v1.5-7b-pretrain \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --save_total_limit 5 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --run_name pretrain-ca \
#     --diversity_weight 1 \
#     --use_ca True \
#     --tune_ca True \
#     --use_router False \
#     --tune_router False \

/home/data/shika/miniconda3/envs/llava/bin/deepspeed --master_port 29501 --include localhost:0,1,2,3 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /home/data/shika/models/lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /home/data/shika/LLaVA/playground/data/LLaVA-Tuning/llava_v1_5_mix65k.json \
    --image_folder /home/data/shika/LLaVA/playground/data/LLaVA-Tuning/ \
    --vision_tower /home/data/shika/models/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/$SPLIT/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/$SPLIT/llava-v1.5-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 20 \
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
    --run_name finetune \
    --diversity_weight 1 \
    --use_ca True \
    --tune_ca True \
    --use_router True \
    --tune_router True 


/home/data/shika/miniconda3/envs/llava/bin/python ./scripts/merge_lora_weights.py \
    --model-path ./checkpoints/$SPLIT/llava-v1.5-7b-pretrain \
    --model-base /home/data/shika/models/lmsys/vicuna-7b-v1.5 \
    --save-model-path ./checkpoints/$SPLIT/llava-v1.5-7b-pretrain-full

/home/data/shika/miniconda3/envs/llava/bin/python ./scripts/merge_lora_weights.py \
    --model-path ./checkpoints/$SPLIT/llava-v1.5-7b-lora \
    --model-base ./checkpoints/$SPLIT/llava-v1.5-7b-pretrain-full \
    --save-model-path ./checkpoints/$SPLIT/llava-v1.5-7b