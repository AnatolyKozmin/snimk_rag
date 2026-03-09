#!/usr/bin/env python3
"""
Расширение FAQ с помощью LLM: генерирует вариации вопросов для каждой пары Q/A.

Использование:
  OPENAI_API_KEY=sk-xxx python scripts/expand_faq_with_llm.py faq.xlsx -o faq_expanded.xlsx
  OPENAI_API_KEY=sk-xxx python scripts/expand_faq_with_llm.py faq.csv -o faq_expanded.csv

Требуется: pip install openai
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.faq_loader import load_faq_from_bytes

PROMPT = """Для каждой пары вопрос-ответ создай {n} альтернативных формулировок вопроса на русском языке.
Формулировки должны быть разговорными, как пользователи реально спрашивают.
Сохраняй смысл, меняй только формулировку.

Вопрос: {question}
Ответ: {answer}

Выведи альтернативные вопросы, по одному на строку, без нумерации."""


def expand_with_openai(question: str, answer: str, api_key: str, n: int = 3) -> list[str]:
    """Генерирует вариации вопроса через OpenAI API."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Ты помощник. Отвечай только списком вопросов, по одному на строку.",
                },
                {"role": "user", "content": PROMPT.format(question=question, answer=answer, n=n)},
            ],
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        if not text:
            return []
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Убрать нумерацию (1., 2., - и т.п.)
        cleaned = []
        for line in lines:
            if line.startswith(("-", "•", "*")):
                line = line[1:].strip()
            elif line and line[0].isdigit() and "." in line[:3]:
                line = line.split(".", 1)[-1].strip()
            if line and line != question:
                cleaned.append(line)
        return cleaned[:n]
    except Exception as e:
        print(f"  Ошибка: {e}", file=sys.stderr)
        return []


def save_to_csv(pairs: list[tuple[str, str]], path: Path) -> None:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Вопрос", "Ответ"])
        writer.writerows(pairs)


def save_to_excel(pairs: list[tuple[str, str]], path: Path) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Вопрос", "Ответ"])
    for q, a in pairs:
        ws.append([q, a])
    wb.save(path)


def main():
    parser = argparse.ArgumentParser(description="Расширение FAQ вариациями вопросов через LLM")
    parser.add_argument("file", type=Path, help="Входной faq.xlsx или faq.csv")
    parser.add_argument("-o", "--output", type=Path, help="Выходной файл (по умолчанию: faq_expanded.xlsx)")
    parser.add_argument("-n", "--variations", type=int, default=3, help="Вар. на вопрос (по умолч. 3)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Задайте OPENAI_API_KEY: export OPENAI_API_KEY=sk-xxx", file=sys.stderr)
        sys.exit(1)

    if not args.file.exists():
        print(f"Файл не найден: {args.file}", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        args.output = args.file.parent / f"{args.file.stem}_expanded{args.file.suffix}"

    pairs = load_faq_from_bytes(args.file.read_bytes(), args.file.name)
    if not pairs:
        print("Нет данных для импорта.", file=sys.stderr)
        sys.exit(1)

    expanded: list[tuple[str, str]] = []
    seen = set()

    for i, (question, answer) in enumerate(pairs):
        expanded.append((question, answer))
        seen.add(question.lower())

        print(f"[{i}/{len(pairs)}] {question[:50]}...")
        variations = expand_with_openai(question, answer, api_key, args.variations)

        for v in variations:
            if v.lower() not in seen:
                expanded.append((v, answer))
                seen.add(v.lower())

    if args.output.suffix.lower() == ".csv":
        save_to_csv(expanded, args.output)
    else:
        save_to_excel(expanded, args.output)

    print(f"\nГотово: {len(pairs)} → {len(expanded)} записей")
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
