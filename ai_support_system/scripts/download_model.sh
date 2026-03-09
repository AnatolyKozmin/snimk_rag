#!/bin/bash
# Загрузка модели через Docker — не требует pip на хосте
# Использование: ./scripts/download_model.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p model_cache

echo "Загрузка модели sentence-transformers/all-MiniLM-L6-v2..."
echo "Кэш: $(pwd)/model_cache"
echo "----------------------------------------"

docker run --rm \
  -v "$(pwd)/model_cache:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  python:3.11-slim \
  sh -c "pip install -q sentence-transformers && python -c \"
from sentence_transformers import SentenceTransformer
print('Загрузка...')
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
m.encode('test')
print('Готово.')
\""

echo "----------------------------------------"
echo "Модель сохранена в ./model_cache"
