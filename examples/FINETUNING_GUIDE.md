# Qwen3-Coder SFT Finetuning Guide

샘플 데이터로 Qwen3-Coder 모델을 파인튜닝하는 단계별 가이드입니다.

## 사전 요구사항

- NVIDIA GPU (최소 24GB VRAM 권장, QLoRA 사용시 20GB 가능)
- CUDA 12.1+
- Conda 설치됨

---

## Quick Start: Qwen3-Coder-30B-A3B QLoRA 학습 (A100 40GB)

30B MoE 모델을 빠르게 파인튜닝하려면 아래 명령어를 순서대로 실행하세요.

```bash
# 1. 환경 설정
conda create -n qwen_sft python=3.9 -y
conda activate qwen_sft
cd /path/to/Qwen3-Coder
pip install -r finetuning/sft/requirements.txt

# 2. 디렉토리 생성
mkdir -p ./data/raw ./data/processed ./checkpoints

# 3. 샘플 데이터 복사
cp examples/sft_data_sample.jsonl ./data/raw/sft_train.jsonl

# 4. 데이터 전처리 (HuggingFace에서 자동 다운로드)
cd finetuning/sft
python binarize_data.py \
    -input_path ../../data/raw/sft_train.jsonl \
    -output_path ../../data/processed/sft_train.jsonl \
    -workers 4 \
    -tokenizer_path Qwen/Qwen3-Coder-30B-A3B-Instruct

# 5. QLoRA 학습 실행
torchrun --nproc_per_node=1 train.py \
    --model_name_or_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --data_path ../../data/processed/sft_train.jsonl \
    --model_max_length 2048 \
    --output_dir ../../checkpoints/qwen3_30b_qlora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_steps 10 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --bf16 True \
    --use_qlora True \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --peft_config_path ./configs/lora
```

---

## Step 1: Conda 환경 생성

```bash
# 새 환경 생성
conda create -n qwen_sft python=3.9 -y
conda activate qwen_sft
```

## Step 2: 의존성 설치

```bash
# 프로젝트 루트로 이동
cd /path/to/Qwen3-Coder

# SFT 의존성 설치
pip install -r finetuning/sft/requirements.txt
```

## Step 3: 모델 다운로드

HuggingFace에서 베이스 모델을 다운로드합니다. 모델은 학습 시 자동으로 다운로드되지만, 미리 받아두려면:

```bash
# huggingface-cli 설치
pip install huggingface_hub

# Qwen3-Coder-30B-A3B-Instruct (권장, MoE 모델)
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct --local-dir ./models/Qwen3-Coder-30B-A3B-Instruct

# 또는 작은 모델로 테스트
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir ./models/Qwen2.5-Coder-1.5B-Instruct
```

## Step 4: 디렉토리 구조 준비

```bash
# 필요한 디렉토리 생성
mkdir -p ./data/raw
mkdir -p ./data/processed
mkdir -p ./checkpoints
```

## Step 5: 샘플 데이터 복사

```bash
# 샘플 데이터를 data 폴더로 복사
cp examples/sft_data_sample.jsonl ./data/raw/sft_train.jsonl
```

## Step 6: 데이터 이진화 (전처리)

```bash
cd finetuning/sft

# 데이터 전처리 실행
python binarize_data.py \
    -input_path ../../data/raw/sft_train.jsonl \
    -output_path ../../data/processed/sft_train.jsonl \
    -workers 4 \
    -tokenizer_path ../../models/Qwen2.5-Coder-1.5B-Instruct
```

## Step 7: 학습 실행

### Option A: LoRA 학습 (권장 - 메모리 효율적)

```bash
cd finetuning/sft

# 환경 변수 설정
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0  # 사용할 GPU 지정

# LoRA 학습 실행
torchrun --nproc_per_node=1 train.py \
    --model_name_or_path ../../models/Qwen2.5-Coder-1.5B-Instruct \
    --data_path ../../data/processed/sft_train.jsonl \
    --model_max_length 1024 \
    --output_dir ../../checkpoints/sft_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_steps 10 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --bf16 True \
    --use_peft True \
    --peft_config_path ./configs/lora
```

### Option B: QLoRA 학습 (4-bit 양자화 - 최소 VRAM)

```bash
cd finetuning/sft

export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0

# QLoRA 학습 실행 (7B 모델도 ~8GB VRAM으로 가능)
torchrun --nproc_per_node=1 train.py \
    --model_name_or_path ../../models/Qwen2.5-Coder-7B-Instruct \
    --data_path ../../data/processed/sft_train.jsonl \
    --model_max_length 1024 \
    --output_dir ../../checkpoints/sft_qlora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_steps 10 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --bf16 True \
    --use_qlora True \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --peft_config_path ./configs/lora
```

### Option C: Full Finetuning (더 많은 VRAM 필요)

```bash
cd finetuning/sft

torchrun --nproc_per_node=1 train.py \
    --model_name_or_path ../../models/Qwen2.5-Coder-1.5B-Instruct \
    --data_path ../../data/processed/sft_train.jsonl \
    --model_max_length 1024 \
    --output_dir ../../checkpoints/sft_full \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 3 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_steps 10 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --deepspeed ./configs/default_offload_opt_param.json \
    --report_to "tensorboard" \
    --bf16 True
```

## Step 8: LoRA 어댑터 병합 (LoRA 사용시)

```bash
cd finetuning/sft

python merge_adapter.py \
    --base_model_path ../../models/Qwen2.5-Coder-1.5B-Instruct \
    --adapter_path ../../checkpoints/sft_lora \
    --output_path ../../checkpoints/sft_merged
```

## Step 9: 학습된 모델 테스트

```python
# test_finetuned_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer

# 병합된 모델 또는 Full finetuning 결과 로드
model_path = "./checkpoints/sft_merged"  # 또는 sft_full

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# 테스트 프롬프트
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Write a Python function to check if a number is prime."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7
)

response = tokenizer.decode(
    generated_ids[0][len(model_inputs.input_ids[0]):],
    skip_special_tokens=True
)
print(response)
```

---

## 파라미터 조정 가이드

| 파라미터 | 설명 | 권장값 |
|----------|------|--------|
| `learning_rate` | 학습률 | LoRA: 1e-4~2e-4, Full: 1e-5~5e-5 |
| `num_train_epochs` | 에폭 수 | 1-5 (데이터 양에 따라) |
| `per_device_train_batch_size` | GPU당 배치 크기 | VRAM에 맞게 조정 (1-8) |
| `gradient_accumulation_steps` | 그래디언트 누적 | 배치 크기 보완용 (4-32) |
| `model_max_length` | 최대 시퀀스 길이 | 512-2048 |
| `lora_r` | LoRA rank | 8-64 (configs/lora/adapter_config.json) |
| `lora_alpha` | LoRA alpha | 16-64 |

## VRAM 요구사항 (추정)

| 모델 크기 | Full FT | LoRA | QLoRA (4-bit) | DeepSpeed ZeRO-3 |
|-----------|---------|------|---------------|------------------|
| 0.5B | ~8GB | ~4GB | ~2GB | ~4GB |
| 1.5B | ~16GB | ~8GB | ~4GB | ~8GB |
| 7B | ~60GB | ~16GB | ~8GB | ~24GB |
| **30B-A3B (MoE)** | ~220GB | ~35GB | **~18GB** | ~70GB |
| 32B | ~240GB | ~40GB | ~20GB | ~80GB |

## 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기 줄이기
--per_device_train_batch_size 1
--gradient_accumulation_steps 32

# 또는 DeepSpeed 사용
--deepspeed ./configs/default_offload_opt_param.json
```

### Flash Attention 오류
```bash
# flash_attn 재설치
pip uninstall flash_attn
pip install flash_attn --no-build-isolation
```

### Multi-GPU 학습
```bash
# GPU 2개 사용
torchrun --nproc_per_node=2 train.py ...

# GPU 지정
export CUDA_VISIBLE_DEVICES=0,1
```

---

## TensorBoard로 학습 모니터링

```bash
tensorboard --logdir ./checkpoints/sft_lora
# 브라우저에서 http://localhost:6006 접속
```
