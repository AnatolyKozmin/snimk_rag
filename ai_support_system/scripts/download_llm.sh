#!/bin/bash
# Загрузка LLM через Docker — не требует pip на хосте
# Использование: ./scripts/download_llm.sh
#
# С токеном Hugging Face: HF_TOKEN=hf_xxx ./scripts/download_llm.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p model_cache

PIP_OPTS="--timeout=120 --retries=5"
[ -n "$PIP_INDEX_URL" ] && PIP_OPTS="$PIP_OPTS -i $PIP_INDEX_URL"

echo "Загрузка LLM Qwen/Qwen2.5-1.5B-Instruct..."
echo "Кэш: $(pwd)/model_cache"
echo "----------------------------------------"

docker run --rm \
  -v "$(pwd)/model_cache:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/root/.cache/huggingface \
  python:3.11-slim \
  sh -c "pip install $PIP_OPTS torch transformers accelerate && python -c \"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('Загрузка токенизатора...')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
print('Загрузка модели...')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-1.5B-Instruct',
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print('Проверка...')
_ = model.generate(**tok('test', return_tensors='pt'), max_new_tokens=5)
print('Готово.')
\""

echo "----------------------------------------"
echo "LLM сохранена в ./model_cache"
