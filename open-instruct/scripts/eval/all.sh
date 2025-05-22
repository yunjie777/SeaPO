#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <GPU_ID> <MODEL>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$1"
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM="false"

MODEL="$2"

MODEL_NAME=$(basename "$MODEL")
echo "Evaluating $MODEL_NAME ..."

python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/$MODEL_NAME \
    --model_name_or_path $MODEL \
    --use_vllm &&

python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/$MODEL_NAME \
    --model_name_or_path $MODEL \
    --use_vllm &&

python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/$MODEL_NAME \
    --model_name_or_path $MODEL \
    --n_shot 8 \
    --use_vllm &&

python -m eval.MATH.run_eval \
    --data_dir data/eval/MATH/ \
    --save_dir results/MATH/$MODEL_NAME \
    --model_name_or_path $MODEL \
    --n_shot 4 \
    --use_vllm &&

export CUDA_VISIBLE_DEVICES=$(echo $1 | cut -d',' -f1)

python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/$MODEL_NAME \
    --model_name_or_path $MODEL \
    --tokenizer_name_or_path $MODEL \
    --eval_batch_size 1 &&

python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/truthfulqa/$MODEL_NAME \
    --model_name_or_path $MODEL \
    --tokenizer_name_or_path $MODEL \
    --metrics mc \
    --preset qa \
    --eval_batch_size 1