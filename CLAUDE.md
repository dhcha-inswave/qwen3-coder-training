# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

Qwen3-Coder는 Alibaba의 코드 생성 대형 언어 모델입니다. 이 저장소는 모델 추론, 파인튜닝(SFT/DPO), 벤치마크 평가를 위한 코드와 스크립트를 제공합니다.

## 주요 디렉토리 구조

- `finetuning/sft/` - Supervised Fine-Tuning 학습 코드
- `finetuning/dpo/` - Direct Preference Optimization 학습 코드
- `qwencoder-eval/` - 평가 벤치마크 스위트
  - `base/` - 베이스 모델 평가
  - `instruct/` - Instruct 모델 평가 (HumanEval, BigCodeBench, Aider 등)
  - `tool_calling_eval/` - 도구 호출 평가
- `demo/` - Gradio 데모 앱
- `examples/` - 사용 예제 코드

## 공통 명령어

### 기본 의존성 설치
```bash
pip install -r requirements.txt
```

### SFT 파인튜닝
```bash
# 환경 설정
conda create -n sft_env python=3.9
conda activate sft_env
pip install -r finetuning/sft/requirements.txt

# 데이터 이진화
bash finetuning/sft/scripts/binarize_data.sh <INPUT_PATH> <OUTPUT_PATH> <TOKENIZER_PATH>

# 학습 실행
bash finetuning/sft/scripts/sft_qwencoder.sh <DATA_PATH> <PRETRAINED_MODEL> <OUTPUT_DIR>

# LoRA 사용시 어댑터 병합
bash finetuning/sft/scripts/merge_adapter.sh <BASE_MODEL_PATH> <ADAPTER_PATH> <OUTPUT_PATH>
```

### DPO 파인튜닝
```bash
conda create -n dpo_env python=3.10
conda activate dpo_env
pip install -r finetuning/dpo/requirements.txt
bash finetuning/dpo/scripts/dpo_qwencoder.sh
```

### 평가 실행
```bash
# Instruct 모델 평가
cd qwencoder-eval/instruct
bash evaluate.sh <MODEL_DIR> <OUTPUT_DIR> <TP>

# Base 모델 평가
cd qwencoder-eval/base
bash run_evaluate_cq2.5.sh
```

### 데모 실행 (Artifacts)
```bash
pip install -r demo/artifacts/requirements.txt
python demo/artifacts/app.py
```

## 모델 사용 패턴

### 기본 추론
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [{"role": "user", "content": "Write a quicksort algorithm"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=65536)
```

### Fill-in-the-Middle (FIM)
```python
prompt = '<|fim_prefix|>' + prefix_code + '<|fim_suffix|>' + suffix_code + '<|fim_middle|>'
```

## 핵심 특징

- 컨텍스트 길이: 256K 토큰 (Yarn으로 1M까지 확장 가능)
- 358개 프로그래밍 언어 지원
- 비생각 모드만 지원 (`<think>` 블록 미생성)
- 전용 도구 파서 필요: `qwen3coder_tool_parser.py`

## 데이터 형식

### SFT 데이터 (JSONL)
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "format": "chatml"}
```

### DPO 데이터 (JSONL)
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```
