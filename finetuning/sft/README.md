# SFT (Supervised Fine-Tuning) for Qwen3-Coder

Qwen3-Coder 모델을 위한 SFT 학습 가이드입니다.

## Quick Start: Qwen3-Coder-30B QLoRA 학습

```bash
# 1. 환경 설정
conda create -n sft_env python=3.9 -y
conda activate sft_env
pip install -r requirements.txt

# 2. 데이터 준비 (raw 폴더에 데이터 넣기)
mkdir -p raw processed

# 3. 데이터 변환 + 토큰화 (한 번에!)
python convert_to_chatml.py ./raw --tokenize

# 4. QLoRA 학습 실행
bash scripts/qlora_qwen3_30b.sh
```

---

## Setup

### Option A: 기본 환경 (QLoRA - Python 3.9)

```bash
conda create -n sft_env python=3.9 -y
conda activate sft_env
pip install -r requirements.txt
```

### Option B: Unsloth 환경 (Python 3.10+ 필수)

Unsloth는 2-5배 빠른 학습을 제공하지만 **Python 3.10 이상**이 필요합니다.

```bash
conda create -n unsloth_env python=3.10 -y
conda activate unsloth_env
pip install -r requirements_py310.txt
```

### 환경 비교

| 환경 | Python | 학습 속도 | 메모리 | requirements |
|-----|--------|----------|--------|--------------|
| `sft_env` | 3.9 | 기본 | 기본 | `requirements.txt` |
| `unsloth_env` | 3.10+ | 2-5배 빠름 | ~70% 절약 | `requirements_py310.txt` |

---

## 데이터 준비

### 지원 입력 형식

`convert_to_chatml.py`를 사용하면 다양한 형식을 자동 변환합니다:

| 형식 | 예시 |
|------|------|
| input/output | `{"input": "질문", "output": "답변"}` |
| instruction/response | `{"instruction": "...", "response": "..."}` |
| question/answer | `{"question": "...", "answer": "..."}` |
| prompt/completion | `{"prompt": "...", "completion": "..."}` |
| messages (ChatML) | 그대로 유지 |

### 데이터 변환 + 토큰화 (권장)

```bash
# 변환 + 토큰화 한 번에 (학습용 데이터 생성)
python convert_to_chatml.py ./raw --tokenize

# 모델과 max_len 지정
python convert_to_chatml.py ./raw --tokenize --model Qwen/Qwen3-Coder-30B-A3B-Instruct --max_len 2048

# 출력 경로 지정
python convert_to_chatml.py ./raw --tokenize -o ./processed/sft_train.jsonl

# system prompt 추가
python convert_to_chatml.py ./raw --tokenize -s "You are a helpful coding assistant."
```

### Train/Eval 데이터 분리

```bash
# 토큰화 + train/eval 자동 분리 (기본 10%)
python convert_to_chatml.py ./raw --tokenize --split

# eval 비율 지정 (20%)
python convert_to_chatml.py ./raw --tokenize --split --eval_ratio 0.2

# 랜덤 시드 지정
python convert_to_chatml.py ./raw --tokenize --split --seed 42
```

**출력 파일:**
- `./processed/sft_train_train.jsonl` - 학습 데이터 (90%)
- `./processed/sft_train_eval.jsonl` - 평가 데이터 (10%)

**학습 시 eval 데이터 사용:**
```bash
EVAL_DATA_PATH="./processed/sft_train_eval.jsonl" bash scripts/qlora_qwen3_30b.sh
```

**토큰화 옵션:**
- `--tokenize`, `-t`: 토큰화 활성화 (input_ids/label 생성)
- `--model`, `-m`: 토크나이저 모델 경로 (기본: `Qwen/Qwen3-Coder-30B-A3B-Instruct`)
- `--max_len`: 최대 시퀀스 길이 (기본: 2048)
- `--split`: train/eval 분리 활성화
- `--eval_ratio`: eval 데이터 비율 (기본: 0.1)
- `--seed`: 랜덤 시드 (기본: 42)

### 데이터 변환만 (ChatML 형식)

```bash
# ChatML 형식으로만 변환 (토큰화 없음)
python convert_to_chatml.py ./raw

# 출력 경로 지정
python convert_to_chatml.py ./raw -o ./processed/sft_train.jsonl

# 기존 방식도 지원
python convert_to_chatml.py -input_dir ./raw -output_dir ./raw_converted -merge
```

**특징:**
- 하위 폴더까지 재귀 검색
- 모든 파일을 하나로 병합
- 기본 출력: `./processed/sft_train.jsonl`

### 데이터 형식

**입력 (raw/):** 다양한 형식 지원
```json
{"input": "질문", "output": "답변"}
{"instruction": "지시", "response": "응답"}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**출력 (processed/):** 토큰화된 형식 (train.py가 사용)
```json
{"input_ids": [1, 2, 3, ...], "label": [-100, -100, 3, ...]}
```

---

## 학습 실행

### Option A: QLoRA 학습 (권장 - 최소 VRAM)

```bash
# 기본 실행 (wandb 자동 설정)
bash scripts/qlora_qwen3_30b.sh

# 커스텀 설정
bash scripts/qlora_qwen3_30b.sh ./processed/sft_train.jsonl ../../checkpoints/my_model my_experiment

# 실험 이름 변경
EXPERIMENT_NAME="my_experiment" bash scripts/qlora_qwen3_30b.sh

# wandb 프로젝트 변경
WANDB_PROJECT="my-project" bash scripts/qlora_qwen3_30b.sh
```

#### QLoRA 스크립트 기본 설정

| 카테고리 | 파라미터 | 기본값 | 설명 |
|---------|---------|--------|------|
| **모델** | `MODEL_NAME` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 베이스 모델 |
| **데이터** | `DATA_PATH` | `./processed/sft_train.jsonl` | 학습 데이터 경로 |
| | `EVAL_DATA_PATH` | `(none)` | 평가 데이터 경로 |
| | `MAX_LENGTH` | `2048` | 최대 시퀀스 길이 |
| **평가** | `EVAL_STEPS` | `50` | 평가 간격 (steps) |
| **학습** | `BATCH_SIZE` | `1` | 배치 크기 |
| | `GRAD_ACCU` | `16` | Gradient Accumulation Steps |
| | `LR` | `1e-4` | Learning Rate |
| | `EPOCHS` | `3` | 학습 에폭 |
| | `WARMUP_STEPS` | `10` | Warmup Steps |
| **QLoRA** | `bnb_4bit_quant_type` | `nf4` | 4비트 양자화 타입 |
| | `bnb_4bit_compute_dtype` | `bfloat16` | 연산 dtype |
| **메모리 최적화** | `gradient_checkpointing` | `True` | 메모리 절약 |
| | `optim` | `paged_adamw_8bit` | 8비트 옵티마이저 |
| **저장** | `save_steps` | `50` | 체크포인트 저장 간격 |
| | `save_total_limit` | `3` | 최대 체크포인트 수 |
| **로깅** | `logging_steps` | `10` | 로그 출력 간격 |
| | `report_to` | `wandb` | 로깅 대상 |

#### 환경 변수로 설정 변경

```bash
# 실험 이름 (OUTPUT_DIR, RUN_NAME에 사용)
EXPERIMENT_NAME="my_exp" bash scripts/qlora_qwen3_30b.sh

# 평가 데이터 경로 지정
EVAL_DATA_PATH="./processed/sft_train_eval.jsonl" bash scripts/qlora_qwen3_30b.sh

# 평가 간격 변경
EVAL_STEPS=100 EVAL_DATA_PATH="./processed/sft_train_eval.jsonl" bash scripts/qlora_qwen3_30b.sh

# wandb 프로젝트
WANDB_PROJECT="my-project" bash scripts/qlora_qwen3_30b.sh

# wandb 오프라인 모드
WANDB_MODE="offline" bash scripts/qlora_qwen3_30b.sh
```

#### 출력 경로

실행 시 타임스탬프가 자동으로 추가됩니다:
```
checkpoints/{EXPERIMENT_NAME}_{TIMESTAMP}/
예: checkpoints/qwen3_30b_qlora_20241215_183000/
```

**VRAM 요구사항:**
| 모델 | QLoRA | LoRA | Full FT |
|------|-------|------|---------|
| 7B | ~8GB | ~16GB | ~60GB |
| 30B-A3B | ~18GB | ~35GB | ~220GB |

### Option B: Unsloth QLoRA 학습 (2-5배 빠름)

Unsloth는 2-5배 빠른 학습 속도와 70% 적은 메모리 사용을 제공합니다.

```bash
# 설치 (최초 1회)
pip install unsloth

# 기본 실행
bash scripts/unsloth_qwen3_30b.sh

# eval 데이터 포함
EVAL_DATA_PATH="./processed/sft_train_eval.jsonl" bash scripts/unsloth_qwen3_30b.sh

# 배치 크기 조정 (GPU 메모리에 따라)
BATCH_SIZE=4 bash scripts/unsloth_qwen3_30b.sh
```

#### Unsloth 스크립트 기본 설정

| 카테고리 | 파라미터 | 기본값 | 설명 |
|---------|---------|--------|------|
| **모델** | `MODEL_NAME` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 베이스 모델 |
| **데이터** | `DATA_PATH` | `./processed/sft_train.jsonl` | 학습 데이터 경로 |
| | `MAX_SEQ_LENGTH` | `2048` | 최대 시퀀스 길이 |
| **학습** | `BATCH_SIZE` | `2` | 배치 크기 |
| | `GRAD_ACCUM` | `8` | Gradient Accumulation Steps |
| | `LR` | `2e-4` | Learning Rate |
| | `EPOCHS` | `3` | 학습 에폭 |
| **LoRA** | `LORA_R` | `16` | LoRA rank |
| | `LORA_ALPHA` | `32` | LoRA alpha |
| | `LORA_DROPOUT` | `0.05` | LoRA dropout |
| **로깅** | `logging_steps` | `10` | 로그 출력 간격 |

#### QLoRA vs Unsloth 비교

| 항목 | QLoRA (bitsandbytes) | Unsloth |
|-----|---------------------|---------|
| 학습 속도 | 기본 | 2-5배 빠름 |
| 메모리 사용 | 기본 | ~70% 절약 |
| 설치 | `pip install bitsandbytes` | `pip install unsloth` |
| GPU 지원 | 모든 NVIDIA GPU | NVIDIA GPU (최적화) |

### Option C: LoRA 학습

```bash
bash scripts/sft_qwencoder_with_lora.sh \
    ./processed/sft_train.jsonl \
    Qwen/Qwen3-Coder-30B-A3B-Instruct \
    ../../checkpoints/lora_model
```

### Option D: Full Fine-tuning

```bash
bash scripts/sft_qwencoder.sh \
    ./processed/sft_train.jsonl \
    Qwen/Qwen3-Coder-30B-A3B-Instruct \
    ../../checkpoints/full_model
```

---

## 어댑터 병합 (LoRA/QLoRA 사용 시)

```bash
python merge_adapter.py \
    --base_model_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --adapter_path ../../checkpoints/qwen3_30b_qlora \
    --output_path ../../checkpoints/merged_model
```

---

## 디렉토리 구조

```
finetuning/sft/
├── raw/                    # 원본 데이터 (다양한 형식)
├── processed/              # 토큰화된 학습 데이터
│   └── sft_train.jsonl     # {"input_ids": [...], "label": [...]}
├── configs/
│   ├── lora/              # LoRA 설정
│   └── default_offload_opt_param.json
├── scripts/
│   ├── qlora_qwen3_30b.sh # QLoRA 학습 스크립트
│   ├── sft_qwencoder.sh   # Full FT 스크립트
│   └── sft_qwencoder_with_lora.sh
├── convert_to_chatml.py   # 데이터 변환 + 토큰화
├── binarize_data.py       # 데이터 전처리 (멀티프로세스)
├── train.py               # 학습 메인
├── merge_adapter.py       # 어댑터 병합
└── requirements.txt
```

---

## 데이터 처리 흐름

### 기본 (train만)
```
raw/*.jsonl (다양한 형식)
       ↓
python convert_to_chatml.py ./raw --tokenize
       ↓
processed/sft_train.jsonl (토큰화됨)
       ↓
bash scripts/qlora_qwen3_30b.sh
       ↓
checkpoints/qwen3_30b_qlora_YYYYMMDD_HHMMSS/
```

### Train/Eval 분리
```
raw/*.jsonl (다양한 형식)
       ↓
python convert_to_chatml.py ./raw --tokenize --split
       ↓
processed/sft_train_train.jsonl (90%)
processed/sft_train_eval.jsonl (10%)
       ↓
EVAL_DATA_PATH="./processed/sft_train_eval.jsonl" bash scripts/qlora_qwen3_30b.sh
       ↓
checkpoints/qwen3_30b_qlora_YYYYMMDD_HHMMSS/
```

---

## 학습 모니터링

### wandb (온라인)
스크립트 실행 시 자동으로 wandb에 로그인하고 메트릭을 기록합니다.
- 대시보드: https://wandb.ai

### TensorBoard (로컬)
```bash
tensorboard --logdir ../../checkpoints/qwen3_30b_qlora
# http://localhost:6006 접속
```

---

## 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기 줄이기 (스크립트 내 BATCH_SIZE 수정)
BATCH_SIZE=1
GRAD_ACCU=32
```

### No space left on device
스크립트가 자동으로 캐시를 프로젝트 `.cache/` 폴더에 저장합니다.
```bash
# 기존 캐시 정리
rm -rf ~/.cache/huggingface
```

### bitsandbytes not found
```bash
pip install bitsandbytes>=0.41.0
```

### target_modules 오류
`configs/lora/adapter_config.json`에서 `target_modules`가 설정되어 있는지 확인:
```json
"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### KeyError: 'input_ids'
데이터가 토큰화되지 않았습니다. `--tokenize` 옵션을 사용하세요:
```bash
python convert_to_chatml.py ./raw --tokenize
```
