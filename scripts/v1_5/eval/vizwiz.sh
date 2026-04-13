#!/bin/bash

/home/data/shika/miniconda3/envs/llava/bin/python -m llava.eval.model_vqa_loader \
    --model-path /path/to/your_checkpoint_dir \
    --question-file /dataset/eval/vizwiz/llava_test.jsonl \
    --image-folder /dataset/eval/vizwiz/test \
    --answers-file /dataset/eval/vizwiz/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

/home/data/shika/miniconda3/envs/llava/bin/python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /dataset/eval/vizwiz/llava_test.jsonl \
    --result-file /dataset/eval/vizwiz/answers/llava-v1.5-7b.jsonl \
    --result-upload-file /dataset/eval/vizwiz/answers_upload/llava-v1.5-7b.json
