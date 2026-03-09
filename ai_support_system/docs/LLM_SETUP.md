# Настройка LLM (Qwen2.5-1.5B-Instruct)

## Что делает LLM

- **RAG-ответы**: вместо дословного ответа из FAQ — естественная формулировка на основе контекста
- **Fallback**: при средней уверенности (score 0.6–0.8) пытается ответить по нескольким FAQ
- **Без галлюцинаций**: модель опирается только на переданный контекст

## Требования

- **RAM**: 6 GB (Docker limit увеличен до 6G)
- **CPU**: любой (модель работает на CPU)
- **GPU**: опционально (если есть — будет быстрее)

## Шаг 1: Загрузка моделей

Перед первым запуском загрузите обе модели (эмбеддинги + LLM):

```bash
cd ai_support_system

# 1. Эмбеддинги (если ещё не загружены)
python scripts/download_model.py

# 2. LLM
python scripts/download_llm.py
```

С токеном Hugging Face (рекомендуется, быстрее):

```bash
HF_TOKEN=hf_xxxxxxxx python scripts/download_model.py
HF_TOKEN=hf_xxxxxxxx python scripts/download_llm.py
```

Модели сохраняются в `./model_cache/`.

## Шаг 2: Запуск

### Локально

```bash
python main.py
```

### Docker

```bash
docker compose up -d
```

## Конфигурация (.env)

```env
USE_LLM_RAG=true
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
LLM_TOP_K=3
SIMILARITY_THRESHOLD=0.8
SIMILARITY_THRESHOLD_LLM=0.6
```

- `USE_LLM_RAG=false` — отключить LLM, работать как раньше (только поиск по FAQ)
- `LLM_TOP_K` — сколько FAQ передавать в контекст LLM (3–5)

## Отключение LLM

Если нужно вернуться к режиму без LLM:

```env
USE_LLM_RAG=false
```

Или в `docker-compose.yml`:

```yaml
- USE_LLM_RAG=false
```

## Развёртывание на сервере

1. Скопируйте `model_cache` на сервер (или загрузите модели там):

```bash
# Вариант A: rsync (надёжнее при нестабильной сети)
rsync -avz --progress ./model_cache/ user@server:/path/to/ai_support_system/model_cache/

# Вариант B: загрузка на сервере
ssh user@server
cd /path/to/ai_support_system
HF_TOKEN=hf_xxx python scripts/download_llm.py
```

2. Увеличьте лимит памяти Docker до 6G (уже в docker-compose).

3. Запустите `docker compose up -d`.
