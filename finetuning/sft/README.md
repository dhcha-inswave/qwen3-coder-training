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

# 3. 데이터 변환 (다양한 형식 → ChatML) - 폴더만 지정하면 끝!
python convert_to_chatml.py ./raw

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

### 데이터 변환

```bash
# 간단하게 폴더만 지정 (권장)
python convert_to_chatml.py ./raw

# 출력 경로 지정
python convert_to_chatml.py ./raw -o ./processed/sft_train.jsonl

# system prompt 추가
python convert_to_chatml.py ./raw -s "You are a helpful coding assistant."

# 기존 방식도 지원
python convert_to_chatml.py -input_dir ./raw -output_dir ./raw_converted -merge
```

**특징:**
- 하위 폴더까지 재귀 검색
- 모든 파일을 하나로 병합
- 기본 출력: `./processed/sft_train.jsonl`

### ChatML 형식 (최종 형식)

```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a regex expression to match any letter"},
        {"role": "assistant", "content": "The regex expression is: [a-zA-Z]"}
    ]
}
```

**참고:**
- `system` role은 선택사항
- `"format": "chatml"` 필드는 필요 없음
- 한 줄에 하나의 JSON 객체 (JSONL 형식)

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
├── raw_converted/          # ChatML로 변환된 데이터
├── processed/              # 최종 학습 데이터
├── configs/
│   ├── lora/              # LoRA 설정
│   └── default_offload_opt_param.json
├── scripts/
│   ├── qlora_qwen3_30b.sh # QLoRA 학습 스크립트
│   ├── sft_qwencoder.sh   # Full FT 스크립트
│   └── sft_qwencoder_with_lora.sh
├── convert_to_chatml.py   # 데이터 변환 스크립트
├── binarize_data.py       # 데이터 전처리
├── train.py               # 학습 메인
├── merge_adapter.py       # 어댑터 병합
└── requirements.txt
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
