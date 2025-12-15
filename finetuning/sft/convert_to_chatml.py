#!/usr/bin/env python3
"""
다양한 형식의 JSON/JSONL 파일을 ChatML 형식으로 변환 + 토큰화하는 스크립트

지원 입력 형식:
1. {"input": "...", "output": "..."} - input/output 형식
2. {"instruction": "...", "response": "..."} - instruction/response 형식
3. {"question": "...", "answer": "..."} - question/answer 형식
4. {"prompt": "...", "completion": "..."} - prompt/completion 형식
5. {"messages": [...]} - 이미 ChatML 형식 (그대로 유지)

출력 형식 (--tokenize 옵션 사용 시):
{"input_ids": [...], "label": [...]}

사용법:
    # 폴더 하나로 간단하게 변환 (권장)
    python convert_to_chatml.py ./raw

    # 토큰화까지 한번에 (학습용 데이터 생성)
    python convert_to_chatml.py ./raw --tokenize --model Qwen/Qwen3-Coder-30B-A3B-Instruct

    # 출력 경로 지정
    python convert_to_chatml.py ./raw -o ./processed/sft_train.jsonl

    # system prompt 추가
    python convert_to_chatml.py ./raw -o ./processed/sft_train.jsonl -s "You are a helpful assistant."

    # 기존 방식도 지원
    python convert_to_chatml.py -input_dir ./raw -output_dir ./raw_converted -merge
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import glob

IGNORE_INDEX = -100


def detect_format(obj: Dict) -> str:
    """JSON 객체의 형식을 감지"""
    if "messages" in obj:
        return "chatml"
    elif "input" in obj and "output" in obj:
        return "input_output"
    elif "instruction" in obj and "response" in obj:
        return "instruction_response"
    elif "instruction" in obj and "output" in obj:
        return "instruction_output"
    elif "question" in obj and "answer" in obj:
        return "question_answer"
    elif "prompt" in obj and "completion" in obj:
        return "prompt_completion"
    elif "prompt" in obj and "response" in obj:
        return "prompt_response"
    elif "query" in obj and "response" in obj:
        return "query_response"
    else:
        return "unknown"


def convert_to_chatml(obj: Dict, system_prompt: Optional[str] = None) -> Optional[Dict]:
    """다양한 형식을 ChatML 형식으로 변환"""

    format_type = detect_format(obj)

    if format_type == "chatml":
        # 이미 ChatML 형식
        return obj

    # 입력/출력 필드 매핑
    field_mappings = {
        "input_output": ("input", "output"),
        "instruction_response": ("instruction", "response"),
        "instruction_output": ("instruction", "output"),
        "question_answer": ("question", "answer"),
        "prompt_completion": ("prompt", "completion"),
        "prompt_response": ("prompt", "response"),
        "query_response": ("query", "response"),
    }

    if format_type == "unknown":
        print(f"Warning: Unknown format, keys: {list(obj.keys())}")
        return None

    input_field, output_field = field_mappings[format_type]
    user_content = obj.get(input_field, "").strip()
    assistant_content = obj.get(output_field, "").strip()

    if not user_content or not assistant_content:
        return None

    messages = []

    # system prompt 추가 (옵션)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})

    return {"messages": messages}


def tokenize_chatml(messages: List[Dict], tokenizer, max_len: int = 2048,
                    system_message: str = "You are a helpful assistant.") -> Optional[Dict]:
    """ChatML 형식의 messages를 토큰화하여 input_ids와 label 생성"""

    im_start = tokenizer("<|im_start|>").input_ids[0]
    im_end = tokenizer("<|im_end|>").input_ids[0]
    nl_tokens = tokenizer('\n').input_ids
    if len(nl_tokens) > 0:
        nl_tokens = nl_tokens[-1:]

    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    input_id, target = [], []

    # system message 처리
    if messages[0]["role"] == "system":
        system_message = messages[0]["content"]
        messages = messages[1:]

    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens

    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    for sentence in messages:
        role = roles.get(sentence["role"])
        if not role:
            continue

        _input_id = tokenizer(role).input_ids + nl_tokens + \
                    tokenizer(sentence["content"], add_special_tokens=False).input_ids + [im_end] + nl_tokens
        input_id += _input_id

        if role == '<|im_start|>user':
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == '<|im_start|>assistant':
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + \
                      _input_id[len(tokenizer(role).input_ids) + 1: -2] + [im_end] + nl_tokens
        else:
            continue

        target += _target

    if len(input_id) > max_len:
        return None

    return {"input_ids": input_id, "label": target}


def process_file(input_path: str, output_path: str, system_prompt: Optional[str] = None) -> Dict:
    """단일 파일 처리"""

    converted_count = 0
    skipped_count = 0
    format_counts = {}

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: JSON parse error - {e}")
                skipped_count += 1
                continue

            # 형식 감지 및 통계
            format_type = detect_format(obj)
            format_counts[format_type] = format_counts.get(format_type, 0) + 1

            # 변환
            converted = convert_to_chatml(obj, system_prompt)

            if converted:
                outfile.write(json.dumps(converted, ensure_ascii=False) + '\n')
                converted_count += 1
            else:
                skipped_count += 1

    return {
        "converted": converted_count,
        "skipped": skipped_count,
        "formats": format_counts
    }


def process_directory(input_dir: str, output_dir: str, system_prompt: Optional[str] = None) -> None:
    """디렉토리 내 모든 JSONL/JSON 파일 처리"""

    # 지원 확장자
    extensions = ['*.jsonl', '*.json']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not files:
        print(f"No JSON/JSONL files found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    total_converted = 0
    total_skipped = 0

    print(f"\n{'='*60}")
    print(f"Processing {len(files)} files from {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        output_filename = filename.replace('.json', '.jsonl') if filename.endswith('.json') else filename
        output_path = os.path.join(output_dir, output_filename)

        print(f"Processing: {filename}")

        result = process_file(filepath, output_path, system_prompt)

        print(f"  Converted: {result['converted']}, Skipped: {result['skipped']}")
        print(f"  Formats detected: {result['formats']}")

        total_converted += result['converted']
        total_skipped += result['skipped']

    print(f"\n{'='*60}")
    print(f"Total: {total_converted} converted, {total_skipped} skipped")
    print(f"{'='*60}\n")


def merge_files(input_dir: str, output_path: str) -> None:
    """디렉토리 내 모든 JSONL 파일을 하나로 병합"""

    files = glob.glob(os.path.join(input_dir, '*.jsonl'))

    if not files:
        print(f"No JSONL files found in {input_dir}")
        return

    total_lines = 0

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for filepath in sorted(files):
            with open(filepath, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if line.strip():
                        outfile.write(line)
                        total_lines += 1

    print(f"Merged {len(files)} files into {output_path}")
    print(f"Total lines: {total_lines}")


def convert_directory_to_single_file(input_dir: str, output_path: str, system_prompt: Optional[str] = None) -> None:
    """디렉토리 내 모든 파일을 변환하여 하나의 파일로 출력"""

    # 지원 확장자
    extensions = ['*.jsonl', '*.json']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        # 하위 폴더도 검색
        files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

    # 중복 제거
    files = list(set(files))

    if not files:
        print(f"No JSON/JSONL files found in {input_dir}")
        return

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    total_converted = 0
    total_skipped = 0
    all_formats = {}

    print(f"\n{'='*60}")
    print(f"Converting {len(files)} files from: {input_dir}")
    print(f"Output: {output_path}")
    if system_prompt:
        print(f"System prompt: {system_prompt[:50]}...")
    print(f"{'='*60}\n")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for filepath in sorted(files):
            filename = os.path.relpath(filepath, input_dir)
            converted_count = 0
            skipped_count = 0

            with open(filepath, 'r', encoding='utf-8') as infile:
                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        skipped_count += 1
                        continue

                    # 형식 통계
                    format_type = detect_format(obj)
                    all_formats[format_type] = all_formats.get(format_type, 0) + 1

                    # 변환
                    converted = convert_to_chatml(obj, system_prompt)

                    if converted:
                        outfile.write(json.dumps(converted, ensure_ascii=False) + '\n')
                        converted_count += 1
                    else:
                        skipped_count += 1

            print(f"  {filename}: {converted_count} converted, {skipped_count} skipped")
            total_converted += converted_count
            total_skipped += skipped_count

    print(f"\n{'='*60}")
    print(f"Total: {total_converted} converted, {total_skipped} skipped")
    print(f"Formats detected: {all_formats}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")

    return output_path


def convert_and_tokenize(input_dir: str, output_path: str, model_path: str,
                         max_len: int = 2048, system_prompt: Optional[str] = None) -> None:
    """디렉토리 내 모든 파일을 변환 + 토큰화하여 하나의 파일로 출력"""

    try:
        import transformers
    except ImportError:
        print("ERROR: transformers가 설치되어 있지 않습니다.")
        print("pip install transformers 실행 후 다시 시도하세요.")
        return

    # 캐시 디렉토리 설정 (프로젝트 .cache 폴더 사용)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    cache_dir = os.path.join(project_root, ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)

    # 환경변수 설정
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    print(f"\n{'='*60}")
    print(f"Loading tokenizer from: {model_path}")
    print(f"Cache directory: {cache_dir}")
    print(f"{'='*60}\n")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=False,
        add_bos_token=False,
        pad_token='<|endoftext|>',
        eos_token='<|im_end|>',
        cache_dir=cache_dir,
        model_max_length=max_len * 5,
        truncation=True,
        padding_side="right",
        trust_remote_code=True
    )
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|repo_name|>",
            "<|file_sep|>", "<|im_start|>", "<|im_end|>"
        ]
    })

    # 지원 확장자
    extensions = ['*.jsonl', '*.json']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

    files = list(set(files))

    if not files:
        print(f"No JSON/JSONL files found in {input_dir}")
        return

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    total_converted = 0
    total_skipped = 0

    print(f"\n{'='*60}")
    print(f"Converting + Tokenizing {len(files)} files from: {input_dir}")
    print(f"Output: {output_path}")
    print(f"Max length: {max_len}")
    print(f"{'='*60}\n")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for filepath in sorted(files):
            filename = os.path.relpath(filepath, input_dir)
            converted_count = 0
            skipped_count = 0

            with open(filepath, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_count += 1
                        continue

                    # ChatML로 변환
                    converted = convert_to_chatml(obj, system_prompt)
                    if not converted:
                        skipped_count += 1
                        continue

                    # 토큰화
                    tokenized = tokenize_chatml(
                        converted["messages"],
                        tokenizer,
                        max_len=max_len,
                        system_message=system_prompt or "You are a helpful assistant."
                    )

                    if tokenized:
                        outfile.write(json.dumps(tokenized, ensure_ascii=False) + '\n')
                        converted_count += 1
                    else:
                        skipped_count += 1

            print(f"  {filename}: {converted_count} converted, {skipped_count} skipped")
            total_converted += converted_count
            total_skipped += skipped_count

    print(f"\n{'='*60}")
    print(f"Total: {total_converted} tokenized, {total_skipped} skipped")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert various JSON formats to ChatML format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 간단하게 폴더 변환 (권장)
  python convert_to_chatml.py ./raw
  python convert_to_chatml.py ./raw -o ./processed/sft_train.jsonl

  # system prompt 추가
  python convert_to_chatml.py ./raw -o ./processed/sft_train.jsonl -s "You are a helpful assistant."

  # 기존 방식 (개별 파일 출력)
  python convert_to_chatml.py -input_dir ./raw -output_dir ./raw_converted -merge
        """
    )

    # 간단한 방식: positional argument
    parser.add_argument('input', nargs='?', type=str, help='Input directory path')
    parser.add_argument('-o', '--output', type=str, default='./processed/sft_train.jsonl', help='Output file path (default: ./processed/sft_train.jsonl)')
    parser.add_argument('-s', '--system', type=str, default=None, help='System prompt to add')

    # 토큰화 옵션
    parser.add_argument('--tokenize', '-t', action='store_true', help='Enable tokenization (produces input_ids/label)')
    parser.add_argument('--model', '-m', type=str, default='Qwen/Qwen3-Coder-30B-A3B-Instruct', help='Model path for tokenizer')
    parser.add_argument('--max_len', type=int, default=2048, help='Maximum sequence length')

    # 기존 방식: named arguments
    parser.add_argument('-input_path', type=str, help='Single input file path')
    parser.add_argument('-output_path', type=str, help='Single output file path')
    parser.add_argument('-input_dir', type=str, help='Input directory containing JSON/JSONL files')
    parser.add_argument('-output_dir', type=str, help='Output directory for converted files')
    parser.add_argument('-system_prompt', type=str, default=None, help='Optional system prompt to add')
    parser.add_argument('-merge', action='store_true', help='Merge all output files into one')
    parser.add_argument('-merge_output', type=str, default='./processed/merged.jsonl', help='Merged output file path')

    args = parser.parse_args()

    # system prompt 통합 (둘 중 하나 사용)
    system_prompt = args.system or args.system_prompt

    # 1. 토큰화 모드: python convert_to_chatml.py ./raw --tokenize
    if args.input and os.path.isdir(args.input) and args.tokenize:
        convert_and_tokenize(args.input, args.output, args.model, args.max_len, system_prompt)

    # 2. 간단한 방식: python convert_to_chatml.py ./raw
    elif args.input and os.path.isdir(args.input):
        convert_directory_to_single_file(args.input, args.output, system_prompt)

    # 2. 단일 파일 처리
    elif args.input_path and args.output_path:
        print(f"Processing single file: {args.input_path}")
        result = process_file(args.input_path, args.output_path, system_prompt)
        print(f"Converted: {result['converted']}, Skipped: {result['skipped']}")
        print(f"Formats: {result['formats']}")
        print(f"Output: {args.output_path}")

    # 3. 디렉토리 처리 (기존 방식)
    elif args.input_dir and args.output_dir:
        process_directory(args.input_dir, args.output_dir, system_prompt)

        # 병합 옵션
        if args.merge:
            merge_files(args.output_dir, args.merge_output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
