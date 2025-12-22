#!/bin/bash
# Unsloth QLoRA Training Script for Qwen3-Coder-30B-A3B-Instruct
#
# Unsloth는 2-5배 빠른 학습 속도와 70% 적은 메모리 사용을 제공합니다.
#
# 요구사항:
#   - NVIDIA GPU (A100 40GB 이상 권장)
#   - CUDA 12.1+
#   - unsloth 설치됨 (pip install unsloth)
#   - wandb 설치됨 (pip install wandb)
#
# 사용법 (finetuning/sft 디렉토리에서 실행):
#   cd finetuning/sft
#   bash scripts/unsloth_qwen3_30b.sh
#
# 예시:
#   EVAL_DATA_PATH="./processed/sft_train_eval.jsonl" bash scripts/unsloth_qwen3_30b.sh

# 스크립트 위치 기준으로 sft 디렉토리로 이동
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SFT_DIR=$(dirname "$SCRIPT_DIR")
cd "$SFT_DIR"
echo "Working directory: $(pwd)"

# 프로젝트 루트 경로
PROJECT_ROOT=$(cd "../.." && pwd)
CACHE_DIR="${PROJECT_ROOT}/.cache"

# 캐시 디렉토리 생성
mkdir -p "${CACHE_DIR}/huggingface"
mkdir -p "${CACHE_DIR}/torch"
mkdir -p "${CACHE_DIR}/wandb"

# 모든 캐시를 프로젝트 폴더로 설정
export HF_HOME="${CACHE_DIR}/huggingface"
export HF_DATASETS_CACHE="${CACHE_DIR}/huggingface/datasets"
export TRANSFORMERS_CACHE="${CACHE_DIR}/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export TORCH_HOME="${CACHE_DIR}/torch"
export WANDB_DIR="${CACHE_DIR}/wandb"
export WANDB_CACHE_DIR="${CACHE_DIR}/wandb"
export XDG_CACHE_HOME="${CACHE_DIR}"

# Triton 캐시 (Unsloth 사용 시 필요)
mkdir -p "${CACHE_DIR}/triton"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"
export TRITON_HOME="${CACHE_DIR}/triton"

# PyTorch Inductor 캐시
mkdir -p "${CACHE_DIR}/torch_inductor"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torch_inductor"

echo "Cache directory: ${CACHE_DIR}"

# train_unsloth.py 존재 확인
if [ ! -f "train_unsloth.py" ]; then
    echo "ERROR: train_unsloth.py not found in $(pwd)"
    exit 1
fi

# 메모리 최적화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 파라미터 설정
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"unsloth_qwen3_30b"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DATA_PATH=${1:-"./processed/sft_train.jsonl"}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-""}
OUTPUT_DIR=${2:-"../../checkpoints/${EXPERIMENT_NAME}_${TIMESTAMP}"}
RUN_NAME=${3:-"${EXPERIMENT_NAME}_${TIMESTAMP}"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-Coder-30B-A3B-Instruct"}

# wandb 설정
export WANDB_PROJECT=${WANDB_PROJECT:-"qwen3-coder-unsloth"}
export WANDB_MODE=${WANDB_MODE:-"online"}

# unsloth 설치 확인
if ! python -c "import unsloth" 2>/dev/null; then
    echo "=========================================="
    echo "unsloth가 설치되어 있지 않습니다. 설치합니다..."
    echo "=========================================="
    pip install unsloth
fi

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

# 학습 하이퍼파라미터
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-2e-4}
EPOCHS=${EPOCHS:-3}
WARMUP_STEPS=${WARMUP_STEPS:-10}

# LoRA 설정
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

# 저장/평가 설정
SAVE_STEPS=${SAVE_STEPS:-50}
EVAL_STEPS=${EVAL_STEPS:-50}
LOGGING_STEPS=${LOGGING_STEPS:-10}

# 체크포인트 재개 설정
RESUME_FROM=${RESUME_FROM:-""}

echo "=========================================="
echo "Unsloth QLoRA Training: Qwen3-Coder-30B"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Train Data: $DATA_PATH"
echo "Eval Data: ${EVAL_DATA_PATH:-"(none)"}"
echo "Output: $OUTPUT_DIR"
echo "Max Seq Length: $MAX_SEQ_LENGTH"
echo "Batch Size: $BATCH_SIZE x $GRAD_ACCUM (grad accum)"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCHS"
echo "LoRA r=$LORA_R, alpha=$LORA_ALPHA"
echo "wandb Project: $WANDB_PROJECT"
echo "wandb Run: $RUN_NAME"
echo "Resume from: ${RESUME_FROM:-\"(none)\"}"
echo "=========================================="

# eval 데이터 인자 설정
EVAL_ARGS=""
if [ -n "$EVAL_DATA_PATH" ] && [ -f "$EVAL_DATA_PATH" ]; then
    EVAL_ARGS="--eval_data_path $EVAL_DATA_PATH --eval_steps $EVAL_STEPS"
fi

# resume 인자 설정
RESUME_ARGS=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARGS="--resume_from_checkpoint $RESUME_FROM"
fi

python train_unsloth.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    $EVAL_ARGS \
    $RESUME_ARGS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --wandb_project $WANDB_PROJECT \
    --run_name $RUN_NAME

echo "=========================================="
echo "Training completed!"
echo "Checkpoint saved to: $OUTPUT_DIR"
echo "=========================================="
