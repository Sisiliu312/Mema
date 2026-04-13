#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m llava.eval.model_vqa_loader \
    --model-path /path/to/your_checkpoint_dir \
    --question-file /dataset/eval/MP-DocVQA/question_val.jsonl \
    --image-folder /dataset/eval/MP-DocVQA/images \
    --answers-file /dataset/eval/MP-DocVQA/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


python -m llava.eval.eval_from_predictions \
    --annotation-file /dataset/eval/MP-DocVQA/val.json \
    --result-file /dataset/eval/MP-DocVQA/answers/llava-v1.5-7b.jsonl
