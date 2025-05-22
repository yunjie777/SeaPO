#!/bin/bash

BASE_MODEL="Qwen2.5-7B-Instruct"
MODEL_NAME="qwen2.5-7b-instruct-kto-demo"

llamafactory-cli train \
    --stage kto \
    --do_train True \
    --model_name_or_path /data/models/$BASE_MODEL \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --dataset kto_en_demo \
    --cutoff_len 2048 \
    --learning_rate 5e-06 \
    --num_train_epochs 3.0 \
    --max_samples 1000000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 5000 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/$BASE_MODEL/lora/$MODEL_NAME \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --pref_beta 0.1 \
    --pref_ftx 0 \
    --pref_loss sigmoid

echo "Training completed. Starting model export..."

llamafactory-cli export \
    --model_name_or_path /data/models/$BASE_MODEL \
    --adapter_name_or_path saves/$BASE_MODEL/lora/$MODEL_NAME \
    --template qwen \
    --finetuning_type lora \
    --trust_remote_code true \
    --export_dir /data/trained_models/$MODEL_NAME \
    --export_size 5 \
    --export_device cpu \
    --export_legacy_format

if [ $? -ne 0 ]; then
    echo "Error occurred during training or export. Exiting..."
    exit 1
fi