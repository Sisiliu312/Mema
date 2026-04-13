

#!/bin/bash
cd /code/LLaVA-scvm-answerloss
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

checkpoint_path="/checkpoints/llava-v1.5-7b-10k-align0.1-lr1e-4/llava-v1.5-7b"

SPLIT_answer="llava-v1.5-7b-10k"
# mme
# python -m llava.eval.model_vqa_loader \
#     --model-path $checkpoint_path \
#     --question-file /dataset/eval/MME/llava_mme.jsonl \
#     --image-folder /dataset/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
#     --answers-file /dataset/eval/MME/answers/${SPLIT_answer}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# cd /dataset/eval/MME/

# python convert_answer_to_mme.py --experiment ${SPLIT_answer}

# cd eval_tool


# python calculation.py --results_dir answers/${SPLIT_answer}




# cd /code/LLaVA-scvm-answerloss
# # textvqa
# python -m llava.eval.model_vqa_loader \
#     --model-path $checkpoint_path \
#     --question-file /dataset/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder /dataset/eval/textvqa/train_images \
#     --answers-file /dataset/eval/textvqa/answers/${SPLIT_answer}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa \
#     --annotation-file /dataset/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file /dataset/eval/textvqa/answers/${SPLIT_answer}.jsonl





# # scienceqa
# python -m llava.eval.model_vqa_science \
#     --model-path $checkpoint_path \
#     --question-file /dataset/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder /dataset/eval/scienceqa/test \
#     --answers-file /dataset/eval/scienceqa/answers/${SPLIT_answer}.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_science_qa.py \
#     --base-dir /dataset/eval/scienceqa \
#     --result-file /dataset/eval/scienceqa/answers/${SPLIT_answer}.jsonl \
#     --output-file /dataset/eval/scienceqa/answers/${SPLIT_answer}_output.jsonl \
#     --output-result /dataset/eval/scienceqa/answers/${SPLIT_answer}_result.json




# # pope
# python -m llava.eval.model_vqa_loader \
#     --model-path $checkpoint_path \
#     --question-file /dataset/eval/pope/llava_pope_test.jsonl \
#     --image-folder /dataset/eval/pope/val2014 \
#     --answers-file /dataset/eval/pope/answers/${SPLIT_answer}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir /dataset/eval/pope/coco \
#     --question-file /dataset/eval/pope/llava_pope_test.jsonl \
#     --result-file /dataset/eval/pope/answers/${SPLIT_answer}.jsonl

# python scripts/v1_5/eval/print.py 




# mmbench_cn
SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path $checkpoint_path \
    --question-file /dataset/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file /dataset/eval/mmbench_cn/answers/$SPLIT/${SPLIT_answer}.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /dataset/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /dataset/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir /dataset/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir /dataset/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment ${SPLIT_answer}



# mmbench
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path $checkpoint_path \
    --question-file /dataset/eval/mmbench/$SPLIT.tsv \
    --answers-file /dataset/eval/mmbench/answers/$SPLIT/${SPLIT_answer}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /dataset/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /dataset/eval/mmbench/$SPLIT.tsv \
    --result-dir /dataset/eval/mmbench/answers/$SPLIT \
    --upload-dir /dataset/eval/mmbench/answers_upload/$SPLIT \
    --experiment ${SPLIT_answer}




# mmvet
python -m llava.eval.model_vqa \
    --model-path $checkpoint_path \
    --question-file /dataset/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /dataset/eval/mm-vet/images \
    --answers-file /dataset/eval/mm-vet/answers/${SPLIT_answer}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /dataset/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /dataset/eval/mm-vet/answers/${SPLIT_answer}.jsonl \
    --dst /dataset/eval/mm-vet/results/${SPLIT_answer}.json





# mmmu
CKPT=${SPLIT_answer}
CONFIG="llava/eval/mmmu/configs/llava1.5.yaml"
OUTPUT_DIR="/dataset/eval/MMMU/answers/$CKPT"
OUTPUT_JSON="$OUTPUT_DIR/merge.json"
ANSWER_DICT="llava/eval/mmmu/answer_dict_val.json"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} python -m llava.eval.mmmu.run_llava \
    --data_path /dataset/eval/MMMU \
    --config_path $CONFIG \
    --model_path $checkpoint_path \
    --output_path "$OUTPUT_JSON" \
    --split "validation" \

python -m llava.eval.mmmu.main_eval_only \
    --output_path "$OUTPUT_JSON" \
    --answer_path "$ANSWER_DICT"




# # seed_bench
# gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT=${SPLIT_answer}

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path $checkpoint_path  \
#         --question-file /dataset/eval/seed_bench/llava-seed-bench.jsonl \
#         --image-folder /dataset/eval/seed_bench \
#         --answers-file /dataset/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=/dataset/eval/seed_bench/answers/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /dataset/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# # Evaluate
# python scripts/convert_seed_for_submission.py \
#     --annotation-file /dataset/eval/seed_bench/SEED-Bench.json \
#     --result-file $output_file \
#     --result-upload-file /dataset/eval/seed_bench/answers_upload/${SPLIT_answer}.jsonl




# # gqa
# gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT=${SPLIT_answer}
# SPLIT="llava_gqa_testdev_balanced"
# GQADIR="/dataset/eval/gqa"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path $checkpoint_path \
#         --question-file /dataset/eval/gqa/$SPLIT.jsonl \
#         --image-folder /dataset/eval/gqa/images \
#         --answers-file /dataset/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=/dataset/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /dataset/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

# cd $GQADIR
# python eval.py --tier testdev_balanced




# cd /code/LLaVA-scvm-answerloss
# # vqav2
# gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT=${SPLIT_answer}
# SPLIT="llava_vqav2_mscoco_test-dev2015"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path $checkpoint_path \
#         --question-file  /dataset/eval/vqav2/$SPLIT.jsonl \
#         --image-folder  /dataset/eval/vqav2/test2015 \
#         --answers-file  /dataset/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=/dataset/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /dataset/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# cd /code/LLaVA-scvm-answerloss
# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT




# python -m llava.eval.model_vqa_v7w \
#     --model-path $checkpoint_path \
#     --image-folder /dataset/eval/v7w/images \
#     --question-file /dataset/eval/v7w/_telling_val_mc.jsonl \
#     --answers-file /dataset/eval/v7w/answers/${SPLIT_answer}.jsonl \
#     --mc_out /dataset/eval/v7w/answers/${SPLIT_answer}_mc.json \
#     --temperature 0 \
#     --num_beams 1 \
#     --max_new_tokens 8 \
#     --force_no_sample

# python -m llava.eval.evaluate_v7w \
#     --dataset visual7w-telling \
#     --dataset-root /dataset/eval/v7w \
#     --mode mc \
#     --split val \
#     --results /dataset/eval/v7w/answers/${SPLIT_answer}_mc.json \
#     --verbose 1 \
#     --topk 1











#!/bin/bash
cd /code/LLaVA-scvm-answerloss
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

checkpoint_path="/checkpoints/llava-v1.5-7b-40k-align0.1-lr1e-4/llava-v1.5-7b"

SPLIT_answer="llava-v1.5-7b-40k"
# mme
python -m llava.eval.model_vqa_loader \
    --model-path $checkpoint_path \
    --question-file /dataset/eval/MME/llava_mme.jsonl \
    --image-folder /dataset/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file /dataset/eval/MME/answers/${SPLIT_answer}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /dataset/eval/MME/

python convert_answer_to_mme.py --experiment ${SPLIT_answer}

cd eval_tool


python calculation.py --results_dir answers/${SPLIT_answer}




cd /code/LLaVA-scvm-answerloss
# textvqa
python -m llava.eval.model_vqa_loader \
    --model-path $checkpoint_path \
    --question-file /dataset/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /dataset/eval/textvqa/train_images \
    --answers-file /dataset/eval/textvqa/answers/${SPLIT_answer}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /dataset/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /dataset/eval/textvqa/answers/${SPLIT_answer}.jsonl





# scienceqa
python -m llava.eval.model_vqa_science \
    --model-path $checkpoint_path \
    --question-file /dataset/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /dataset/eval/scienceqa/test \
    --answers-file /dataset/eval/scienceqa/answers/${SPLIT_answer}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /dataset/eval/scienceqa \
    --result-file /dataset/eval/scienceqa/answers/${SPLIT_answer}.jsonl \
    --output-file /dataset/eval/scienceqa/answers/${SPLIT_answer}_output.jsonl \
    --output-result /dataset/eval/scienceqa/answers/${SPLIT_answer}_result.json




# pope
python -m llava.eval.model_vqa_loader \
    --model-path $checkpoint_path \
    --question-file /dataset/eval/pope/llava_pope_test.jsonl \
    --image-folder /dataset/eval/pope/val2014 \
    --answers-file /dataset/eval/pope/answers/${SPLIT_answer}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /dataset/eval/pope/coco \
    --question-file /dataset/eval/pope/llava_pope_test.jsonl \
    --result-file /dataset/eval/pope/answers/${SPLIT_answer}.jsonl

python scripts/v1_5/eval/print.py 




# mmbench_cn
SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path $checkpoint_path \
    --question-file /dataset/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file /dataset/eval/mmbench_cn/answers/$SPLIT/${SPLIT_answer}.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /dataset/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /dataset/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir /dataset/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir /dataset/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment ${SPLIT_answer}



# mmbench
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path $checkpoint_path \
    --question-file /dataset/eval/mmbench/$SPLIT.tsv \
    --answers-file /dataset/eval/mmbench/answers/$SPLIT/${SPLIT_answer}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /dataset/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /dataset/eval/mmbench/$SPLIT.tsv \
    --result-dir /dataset/eval/mmbench/answers/$SPLIT \
    --upload-dir /dataset/eval/mmbench/answers_upload/$SPLIT \
    --experiment ${SPLIT_answer}




# mmvet
python -m llava.eval.model_vqa \
    --model-path $checkpoint_path \
    --question-file /dataset/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /dataset/eval/mm-vet/images \
    --answers-file /dataset/eval/mm-vet/answers/${SPLIT_answer}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /dataset/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /dataset/eval/mm-vet/answers/${SPLIT_answer}.jsonl \
    --dst /dataset/eval/mm-vet/results/${SPLIT_answer}.json





# mmmu
CKPT=${SPLIT_answer}
CONFIG="llava/eval/mmmu/configs/llava1.5.yaml"
OUTPUT_DIR="/dataset/eval/MMMU/answers/$CKPT"
OUTPUT_JSON="$OUTPUT_DIR/merge.json"
ANSWER_DICT="llava/eval/mmmu/answer_dict_val.json"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} python -m llava.eval.mmmu.run_llava \
    --data_path /dataset/eval/MMMU \
    --config_path $CONFIG \
    --model_path $checkpoint_path \
    --output_path "$OUTPUT_JSON" \
    --split "validation" \

python -m llava.eval.mmmu.main_eval_only \
    --output_path "$OUTPUT_JSON" \
    --answer_path "$ANSWER_DICT"




# # seed_bench
# gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT=${SPLIT_answer}

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path $checkpoint_path  \
#         --question-file /dataset/eval/seed_bench/llava-seed-bench.jsonl \
#         --image-folder /dataset/eval/seed_bench \
#         --answers-file /dataset/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=/dataset/eval/seed_bench/answers/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /dataset/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# # Evaluate
# python scripts/convert_seed_for_submission.py \
#     --annotation-file /dataset/eval/seed_bench/SEED-Bench.json \
#     --result-file $output_file \
#     --result-upload-file /dataset/eval/seed_bench/answers_upload/${SPLIT_answer}.jsonl




# # gqa
# gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT=${SPLIT_answer}
# SPLIT="llava_gqa_testdev_balanced"
# GQADIR="/dataset/eval/gqa"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path $checkpoint_path \
#         --question-file /dataset/eval/gqa/$SPLIT.jsonl \
#         --image-folder /dataset/eval/gqa/images \
#         --answers-file /dataset/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=/dataset/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /dataset/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

# cd $GQADIR
# python eval.py --tier testdev_balanced




# cd /code/LLaVA-scvm-answerloss
# # vqav2
# gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT=${SPLIT_answer}
# SPLIT="llava_vqav2_mscoco_test-dev2015"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path $checkpoint_path \
#         --question-file  /dataset/eval/vqav2/$SPLIT.jsonl \
#         --image-folder  /dataset/eval/vqav2/test2015 \
#         --answers-file  /dataset/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=/dataset/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /dataset/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# cd /code/LLaVA-scvm-answerloss
# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT




# python -m llava.eval.model_vqa_v7w \
#     --model-path $checkpoint_path \
#     --image-folder /dataset/eval/v7w/images \
#     --question-file /dataset/eval/v7w/_telling_val_mc.jsonl \
#     --answers-file /dataset/eval/v7w/answers/${SPLIT_answer}.jsonl \
#     --mc_out /dataset/eval/v7w/answers/${SPLIT_answer}_mc.json \
#     --temperature 0 \
#     --num_beams 1 \
#     --max_new_tokens 8 \
#     --force_no_sample

# python -m llava.eval.evaluate_v7w \
#     --dataset visual7w-telling \
#     --dataset-root /dataset/eval/v7w \
#     --mode mc \
#     --split val \
#     --results /dataset/eval/v7w/answers/${SPLIT_answer}_mc.json \
#     --verbose 1 \
#     --topk 1




