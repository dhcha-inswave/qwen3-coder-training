#!/bin/bash
# QLoRA Training Script for Qwen3-Coder-30B-A3B-Instruct
#
# 요구사항:
#   - NVIDIA GPU (A100 40GB 이상 권장)
#   - CUDA 12.1+
#   - bitsandbytes 설치됨
#   - wandb 설치됨 (pip install wandb)
#
# 사용법 (finetuning/sft 디렉토리에서 실행):
#   cd finetuning/sft
#   bash scripts/qlora_qwen3_30b.sh [DATA_PATH] [OUTPUT_DIR] [WANDB_RUN_NAME]
#
# 예시:
#   bash scripts/qlora_qwen3_30b.sh ./processed/sft_train.jsonl ../../checkpoints/qwen3_30b_qlora my_experiment

# 스크립트 위치 기준으로 sft 디렉토리로 이동
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SFT_DIR=$(dirname "$SCRIPT_DIR")
cd "$SFT_DIR"
echo "Working directory: $(pwd)"

# train.py 존재 확인
if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found in $(pwd)"
    echo "Please run from finetuning/sft directory:"
    echo "  cd finetuning/sft && bash scripts/qlora_qwen3_30b.sh"
    exit 1
fi

export NCCL_DEBUG=WARN

# 파라미터 설정
DATA_PATH=${1:-"./processed/sft_train.jsonl"}
OUTPUT_DIR=${2:-"../../checkpoints/qwen3_30b_qlora"}
RUN_NAME=${3:-"qwen3_30b_qlora_$(date +%Y%m%d_%H%M%S)"}
MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# wandb 설정
export WANDB_PROJECT=${WANDB_PROJECT:-"qwen3-coder-finetune"}
export WANDB_MODE=${WANDB_MODE:-"online"}

# wandb 설치 확인
if ! python -c "import wandb" 2>/dev/null; then
    echo "=========================================="
    echo "wandb가 설치되어 있지 않습니다. 설치합니다..."
    echo "=========================================="
    pip install wandb
fi

# wandb 로그인 확인
if ! python -c "import wandb; wandb.api.api_key" 2>/dev/null; then
    echo "=========================================="
    echo "wandb 로그인이 필요합니다."
    echo "https://wandb.ai/authorize 에서 API 키를 복사하세요."
    echo "=========================================="
    python -m wandb login
fi

# GPU 수 자동 감지
GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected GPUs: $GPUS_PER_NODE"

# 학습 하이퍼파라미터
MAX_LENGTH=2048
BATCH_SIZE=1
GRAD_ACCU=16
LR=1e-4
EPOCHS=3
WARMUP_STEPS=10

echo "=========================================="
echo "QLoRA Training: Qwen3-Coder-30B-A3B-Instruct"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Max Length: $MAX_LENGTH"
echo "Batch Size: $BATCH_SIZE x $GRAD_ACCU (grad accum)"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCHS"
echo "wandb Project: $WANDB_PROJECT"
echo "wandb Run: $RUN_NAME"
echo "=========================================="

torchrun --nproc_per_node=$GPUS_PER_NODE train.py \
    --model_name_or_path $MODEL_NAME \
    --data_path $DATA_PATH \
    --model_max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCU \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 3 \
    --learning_rate $LR \
    --weight_decay 0.0 \
    --warmup_steps $WARMUP_STEPS \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --report_to "wandb" \
    --run_name $RUN_NAME \
    --bf16 True \
    --use_qlora True \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --peft_config_path ./configs/lora

echo "=========================================="
echo "Training completed!"
echo "Checkpoint saved to: $OUTPUT_DIR"
echo "=========================================="
