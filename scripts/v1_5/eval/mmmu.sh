#!/bin/bash
CKPT="llava-v1.5-7b"
CONFIG="llava/eval/mmmu/configs/llava1.5.yaml"
OUTPUT_DIR="/dataset/eval/MMMU/answers/$CKPT"
OUTPUT_JSON="$OUTPUT_DIR/merge.json"
ANSWER_DICT="llava/eval/mmmu/answer_dict_val.json"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m llava.eval.mmmu.run_llava \
    --data_path /dataset/eval/MMMU \
    --config_path $CONFIG \
    --model_path /path/to/your_checkpoint_dir \
    --output_path "$OUTPUT_JSON" \
    --split "validation" \

python -m llava.eval.mmmu.main_eval_only \
    --output_path "$OUTPUT_JSON" \
    --answer_path "$ANSWER_DICT"