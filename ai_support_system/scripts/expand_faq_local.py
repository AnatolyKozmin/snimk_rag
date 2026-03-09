#!/usr/bin/env python3
"""
Расширение FAQ с помощью локальной модели Qwen (без API).
Генерирует вариации вопросов для каждой пары Q/A.

Использование:
  python scripts/expand_faq_local.py faq.xlsx -o faq_expanded.csv
  python scripts/expand_faq_local.py faq.csv -o faq_expanded.xlsx
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Кэш моделей
os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent / "model_cache"))

from services.faq_loader import load_faq_from_bytes

EXPAND_PROMPT = """Вопрос: {question}
Ответ: {answer}

Напиши 3 других способа задать этот же вопрос (разговорным языком). Только вопросы, по одному на строку, без нумерации."""


def generate_variations(question: str, answer: str, tokenizer, model, device, n: int = 3) -> list[str]:
    """Генерирует вариации через локальную Qwen."""
    import torch

    messages = [
        {"role": "system", "content": "Ты помощник. Отвечай только списком вопросов, по одному на строку."},
        {"role": "user", "content": EXPAND_PROMPT.format(question=question, answer=answer)},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    lines = [l.strip() for l in response.split("\n") if l.strip()]

    cleaned = []
    for line in lines:
        if line.startswith(("-", "•", "*")):
            line = line[1:].strip()
        elif line and line[0].isdigit() and "." in line[:3]:
            line = line.split(".", 1)[-1].strip()
        if line and line != question and len(line) > 5:
            cleaned.append(line)

    return cleaned[:n]


def main():
    parser = argparse.ArgumentParser(description="Расширение FAQ локальной Qwen")
    parser.add_argument("file", type=Path, help="Входной faq.xlsx или faq.csv")
    parser.add_argument("-o", "--output", type=Path, help="Выходной файл (по умолч. faq_expanded.csv)")
    parser.add_argument("-n", "--variations", type=int, default=3, help="Вар. на вопрос (по умолч. 3)")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Файл не найден: {args.file}", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        args.output = args.file.parent / f"{args.file.stem}_expanded.csv"

    pairs = load_faq_from_bytes(args.file.read_bytes(), args.file.name)
    if not pairs:
        print("Нет данных.", file=sys.stderr)
        sys.exit(1)

    print("Загрузка модели Qwen...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = str(Path(__file__).parent.parent / "model_cache")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True, cache_dir=cache)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        cache_dir=cache,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print("Модель загружена.\n")

    expanded: list[tuple[str, str]] = []
    seen: set[str] = set()

    for i, (question, answer) in enumerate(pairs):
        expanded.append((question, answer))
        seen.add(question.lower())

        print(f"[{i + 1}/{len(pairs)}] {question[:55]}...")
        try:
            variations = generate_variations(question, answer, tokenizer, model, device, args.variations)
            for v in variations:
                if v.lower() not in seen:
                    expanded.append((v, answer))
                    seen.add(v.lower())
        except Exception as e:
            print(f"  Ошибка: {e}", file=sys.stderr)

    # Сохранение
    if args.output.suffix.lower() == ".csv":
        import csv

        with open(args.output, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Вопрос", "Ответ"])
            w.writerows(expanded)
    else:
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Вопрос", "Ответ"])
        for q, a in expanded:
            ws.append([q, a])
        wb.save(args.output)

    print(f"\nГотово: {len(pairs)} → {len(expanded)} записей")
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
