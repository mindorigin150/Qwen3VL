#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
# NPROC_PER_NODE=4
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
# LoRA内存占用小，ZeRO-2通常足够且可能更快
deepspeed=./scripts/zero2.json # 建议为LoRA创建一个zero2.json配置文件

# Model configuration
llm=/run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-VL-7B-Instruct

# Training entry point
entry_file=./qwenvl/train/train_qwen.py

# Dataset configuration
datasets="identity-matching-train%100" # 使用你在 __init__.py 中注册的数据集名称
datasets_val="identity-matching-val%100"

# Output configuration
run_name="qwen2.5vl_7b_lora-r32-a128_identity_matching" # 描述性名称
output_dir=./output/${run_name}

# ============================================
# LoRA 超参数
# ============================================
use_lora=True
lora_rank=32
lora_alpha=128
lora_dropout=0.05
# Qwen3模型，注意力层和部分MLP层是常见的选择
lora_target_modules="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj"

# Training hyperparameters
lr=5e-5  # LoRA可以使用更高的学习率
batch_size=4
grad_accum_steps=4

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --dataset_eval_use ${datasets_val}\
    --data_flatten True \
    
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \

    --use_lora ${use_lora} \
    --lora_r ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_target_modules \"${lora_target_modules}\" \
    
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 8192 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

