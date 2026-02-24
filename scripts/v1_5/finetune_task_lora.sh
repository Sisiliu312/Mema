# !/bin/bash
cd /code/LLaVA-DSU-dynamic-multi-cl-finetune
export PYTHONWARNINGS="ignore"
export PYTHONPATH=/code/LLaVA-DSU-dynamic-multi-cl-finetune
SPLIT="llava-v1.5-dynamic-multi-cl-finetune-loss"

deepspeed --master_port 29400 --include localhost:0,1 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /models/llava-v1.5-7b \
    --version v1 \
    --tune_dsu True \
    --data_path /dataset/LLaVA-Tuning/llava_v1_5_mix20k.json  \
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
    --learning_rate 1e-4 \
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