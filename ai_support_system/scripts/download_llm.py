#!/usr/bin/env python3
"""
Загрузка LLM модели Qwen2.5-1.5B-Instruct.
Запустите перед первым docker compose up — модель сохранится в ./model_cache.

Использование:
  python scripts/download_llm.py

  # С токеном Hugging Face (рекомендуется):
  HF_TOKEN=hf_xxx python scripts/download_llm.py
"""
import os
import sys
from pathlib import Path

# Путь к кэшу — тот же, что монтируется в Docker
CACHE_DIR = Path(__file__).parent.parent / "model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    print(f"Загрузка LLM: {MODEL_NAME}")
    print(f"Кэш: {CACHE_DIR.absolute()}")
    print("—" * 50)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # Проверка
        inputs = tokenizer("Привет", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        _ = model.generate(**inputs, max_new_tokens=5)

        print("—" * 50)
        print("Готово. LLM загружена и сохранена.")
        size_mb = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"Размер кэша: {size_mb:.1f} MB")
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
