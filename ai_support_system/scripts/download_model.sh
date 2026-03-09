#!/bin/bash
# Загрузка модели через Docker — не требует pip на хосте
# Использование: ./scripts/download_model.sh
#
# При медленном интернете можно попробовать зеркало PyPI:
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/download_model.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p model_cache

PIP_OPTS="--timeout=120 --retries=5"
[ -n "$PIP_INDEX_URL" ] && PIP_OPTS="$PIP_OPTS -i $PIP_INDEX_URL"

echo "Загрузка модели sentence-transformers/all-MiniLM-L6-v2..."
echo "Кэш: $(pwd)/model_cache"
echo "----------------------------------------"

docker run --rm \
  -v "$(pwd)/model_cache:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  python:3.11-slim \
  sh -c "pip install $PIP_OPTS sentence-transformers && python -c \"
from sentence_transformers import SentenceTransformer
print('Загрузка...')
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
m.encode('test')
print('Готово.')
\""

echo "----------------------------------------"
echo "Модель сохранена в ./model_cache"
