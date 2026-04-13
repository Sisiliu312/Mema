#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m llava.eval.model_vqa_loader \
    --model-path /path/to/your_checkpoint_dir \
    --question-file /dataset/eval/zerobench/zerobench_llava_questions.jsonl \
    --image-folder /dataset/eval/zerobench \
    --answers-file /dataset/eval/zerobench/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_zerobench \
    --gt-json /dataset/eval/zerobench/zerobench_gt.json \
    --result-file /dataset/eval/zerobench/answers/llava-v1.5-7b-334.jsonl \
    --mode official
