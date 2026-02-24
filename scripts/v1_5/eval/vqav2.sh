cd /code/LLaVA-DSU-dynamic-multi-cl-finetune
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /checkpoints/llava-v1.5-dynamic-multi-cl-finetune-loss/llava-v1.5-7b \
        --question-file  /dataset/eval/vqav2/$SPLIT.jsonl \
        --image-folder  /dataset/eval/vqav2/test2015 \
        --answers-file  /dataset/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/dataset/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /dataset/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

cd /code/LLaVA-DSU-dynamic-multi-cl-finetune
python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

