#!/usr/bin/env python3
"""
Unsloth를 사용한 QLoRA Fine-tuning 스크립트

Unsloth는 2-5배 빠른 학습 속도와 70% 적은 메모리 사용을 제공합니다.

사용법:
    python train_unsloth.py --data_path ./processed/sft_train.jsonl
    python train_unsloth.py --data_path ./processed/sft_train.jsonl --eval_data_path ./processed/sft_train_eval.jsonl
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime

# 캐시 디렉토리 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR

def parse_args():
    parser = argparse.ArgumentParser(description='Unsloth QLoRA Fine-tuning')

    # 데이터
    parser.add_argument('--data_path', type=str, default='./processed/sft_train.jsonl', help='학습 데이터 경로')
    parser.add_argument('--eval_data_path', type=str, default=None, help='평가 데이터 경로')

    # 모델
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-Coder-30B-A3B-Instruct', help='베이스 모델')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='최대 시퀀스 길이')

    # LoRA
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')

    # 학습
    parser.add_argument('--batch_size', type=int, default=2, help='배치 크기')
    parser.add_argument('--grad_accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=3, help='학습 에폭')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=10, help='Warmup steps')

    # 저장
    parser.add_argument('--output_dir', type=str, default=None, help='출력 디렉토리')
    parser.add_argument('--save_steps', type=int, default=50, help='체크포인트 저장 간격')

    # 평가
    parser.add_argument('--eval_steps', type=int, default=50, help='평가 간격')

    # 로깅
    parser.add_argument('--logging_steps', type=int, default=10, help='로깅 간격')
    parser.add_argument('--wandb_project', type=str, default='qwen3-coder-unsloth', help='wandb 프로젝트명')
    parser.add_argument('--run_name', type=str, default=None, help='wandb run 이름')

    return parser.parse_args()


def load_dataset_from_jsonl(file_path):
    """토큰화된 JSONL 파일 로드"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class TokenizedDataset(torch.utils.data.Dataset):
    """이미 토큰화된 데이터셋"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'labels': torch.tensor(item['label'], dtype=torch.long),
        }


def main():
    args = parse_args()

    # 타임스탬프
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 출력 디렉토리 설정
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, "checkpoints", f"unsloth_qlora_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Run name 설정
    if args.run_name is None:
        args.run_name = f"unsloth_qlora_{timestamp}"

    print("=" * 60)
    print("Unsloth QLoRA Fine-tuning")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Train Data: {args.data_path}")
    print(f"Eval Data: {args.eval_data_path or '(none)'}")
    print(f"Output: {args.output_dir}")
    print(f"Max Seq Length: {args.max_seq_length}")
    print(f"Batch Size: {args.batch_size} x {args.grad_accum} (grad accum)")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"wandb Project: {args.wandb_project}")
    print(f"wandb Run: {args.run_name}")
    print("=" * 60)

    # Unsloth 임포트
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
    except ImportError:
        print("ERROR: unsloth가 설치되어 있지 않습니다.")
        print("pip install unsloth 실행 후 다시 시도하세요.")
        sys.exit(1)

    # wandb 설정
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args)
        )
        use_wandb = True
        print("wandb 연결됨")
    except Exception as e:
        print(f"wandb 연결 실패: {e}")
        use_wandb = False

    # 모델 로드
    print("\n모델 로딩 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # 자동 감지
        load_in_4bit=True,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    # LoRA 적용
    print("LoRA 적용 중...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    model.print_trainable_parameters()

    # 데이터셋 로드
    print("\n데이터셋 로딩 중...")
    train_data = load_dataset_from_jsonl(args.data_path)
    train_dataset = TokenizedDataset(train_data)
    print(f"Train 샘플 수: {len(train_dataset)}")

    eval_dataset = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        eval_data = load_dataset_from_jsonl(args.eval_data_path)
        eval_dataset = TokenizedDataset(eval_data)
        print(f"Eval 샘플 수: {len(eval_dataset)}")

    # Trainer 설정
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        seed=42,
        report_to="wandb" if use_wandb else "none",
        run_name=args.run_name,
    )

    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 학습 시작
    print("\n학습 시작...")
    trainer.train()

    # 모델 저장
    print("\n모델 저장 중...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # wandb 종료
    if use_wandb:
        wandb.finish()

    print("=" * 60)
    print("학습 완료!")
    print(f"Checkpoint saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
