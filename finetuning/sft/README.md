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

### 1. Conda 환경 생성

```bash
conda create -n sft_env python=3.9
conda activate sft_env
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

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

**토큰화 옵션:**
- `--tokenize`, `-t`: 토큰화 활성화 (input_ids/label 생성)
- `--model`, `-m`: 토크나이저 모델 경로 (기본: `Qwen/Qwen3-Coder-30B-A3B-Instruct`)
- `--max_len`: 최대 시퀀스 길이 (기본: 2048)

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

# wandb 프로젝트 변경
WANDB_PROJECT="my-project" bash scripts/qlora_qwen3_30b.sh
```

**VRAM 요구사항:**
| 모델 | QLoRA | LoRA | Full FT |
|------|-------|------|---------|
| 7B | ~8GB | ~16GB | ~60GB |
| 30B-A3B | ~18GB | ~35GB | ~220GB |

### Option B: LoRA 학습

```bash
bash scripts/sft_qwencoder_with_lora.sh \
    ./processed/sft_train.jsonl \
    Qwen/Qwen3-Coder-30B-A3B-Instruct \
    ../../checkpoints/lora_model
```

### Option C: Full Fine-tuning

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

```
raw/*.jsonl (다양한 형식)
       ↓
python convert_to_chatml.py ./raw --tokenize
       ↓
processed/sft_train.jsonl (토큰화됨)
       ↓
bash scripts/qlora_qwen3_30b.sh
       ↓
checkpoints/qwen3_30b_qlora (학습된 어댑터)
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
