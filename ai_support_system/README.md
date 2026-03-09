# FAQ Telegram Assistant

Самообучающийся FAQ-ассистент для Telegram. Система автоматически учится на ответах администратора и обновляет векторный индекс.

## Возможности

- **Вопрос-ответ**: пользователи задают вопросы в Telegram, бот ищет ответ в FAQ
- **Самообучение**: неотвеченные вопросы попадают в очередь администратора; после ответа пара Q/A добавляется в базу
- **Кластеризация**: похожие вопросы группируются в кластеры (DBSCAN)
- **Векторный поиск**: FAISS + sentence-transformers (all-MiniLM-L6-v2)
- **Админ-панель**: веб-интерфейс для ответов на pending вопросы

## Быстрый старт (Docker Compose)

Сервисы разделены: **API** (тяжёлый, с моделью) и **бот** (лёгкий). Перезапуск бота — секунды, API остаётся с загруженной моделью.

```bash
# 1. Клонировать/перейти в директорию проекта
cd ai_support_system

# 2. Создать .env с токеном бота
cp .env.example .env
# Отредактировать .env: TELEGRAM_BOT_TOKEN=ваш_токен

# 3. (Рекомендуется) Загрузить модель вручную — быстрее и с контролем
# Вариант A — через Docker (не нужен pip на хосте):
bash scripts/download_model.sh

# Вариант B — через Python (если pip установлен):
pip install sentence-transformers && python scripts/download_model.py

# 4. Запуск обоих сервисов
docker compose up -d --build

# 5. (Опционально) Автоимпорт FAQ при старте
# Положите faq.xlsx в ./data/ или в корень проекта — он импортируется при запуске

# 6. Админ-панель: http://localhost:8009/admin
# 7. API: http://localhost:8009/docs
```

**Перезапуск только бота** (при изменении логики бота):
```bash
docker compose restart faq_bot
```

**Перезапуск только API** (при изменении бэкенда):
```bash
docker compose restart faq_api
```

## Локальный запуск (без Docker)

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Переменные окружения
export TELEGRAM_BOT_TOKEN=ваш_токен
# или создать .env

# 3. Запуск (API + бот)
python main.py

# 4. (Опционально) Начальные FAQ
python scripts/seed_faq.py
```

- **API**: http://localhost:8000
- **Админ-панель**: http://localhost:8000/admin
- **Swagger**: http://localhost:8000/docs

## API Endpoints

### POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Как оплатить заказ?"}'
```

Ответ:
```json
{
  "answer": "Вы можете оплатить картой или наличными при получении.",
  "confidence": 0.92,
  "status": "answered"
}
```

При низкой уверенности (`status: "pending"`):
```json
{
  "answer": "Ваш вопрос передан администратору.",
  "confidence": 0.45,
  "status": "pending"
}
```

### GET /admin/pending

Список ожидающих вопросов с кластерами.

### POST /admin/answer

Добавление ответа (form-data: `question`, `answer`).

### POST /admin/answer/pending/{id}

Ответ на конкретный pending вопрос (form-data: `answer`).

### POST /admin/answer/cluster

Ответ на кластер (JSON: `questions`, `answer`).

### GET /admin/export

Экспорт FAQ в JSON.

### POST /admin/rebuild-index

Пересборка FAISS индекса.

## Структура проекта

```
ai_support_system/
├── api/
│   └── routes.py          # API /ask
├── admin/
│   ├── admin_routes.py    # Админ-панель
│   └── templates/
├── bot/
│   └── telegram_bot.py   # aiogram бот
├── core/
│   └── config.py
├── database/
│   ├── db.py
│   └── models.py
├── services/
│   ├── embedding_service.py
│   ├── faq_service.py
│   ├── clustering_service.py
│   ├── learning_service.py
│   ├── pending_service.py
│   └── normalizer.py
├── vectorstore/
│   └── faiss_index.py
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Требования к серверу

- 4 CPU cores
- 16 GB RAM (рекомендуется)
- Без GPU

Модель эмбеддингов загружается один раз и остаётся в памяти (~80–100 MB).

## Импорт FAQ из Excel / Google Таблиц

**Вариант 1 — через админ-панель:** вкладка «Импорт FAQ» → загрузить файл .xlsx или .csv.

**Вариант 2 — через скрипт:**
```bash
# Импорт (добавить к существующим)
python scripts/import_from_excel.py faq.xlsx

# Импорт с очисткой базы
python scripts/import_from_excel.py faq.xlsx --clear
```

**Формат файла:** колонка 1 = вопрос, колонка 2 = ответ. Первая строка может быть заголовком (вопрос/ответ).

**Экспорт из Google Таблиц:** Файл → Скачать → Microsoft Excel (.xlsx) или CSV.

## Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| TELEGRAM_BOT_TOKEN | Токен бота от @BotFather | — |
| API_HOST | Хост API | 0.0.0.0 |
| API_PORT | Порт API | 8000 |
| DATABASE_URL | URL БД | sqlite+aiosqlite:///./data/faq.db |
| SIMILARITY_THRESHOLD | Порог уверенности | 0.8 |
| CLUSTERING_EPS | DBSCAN eps | 0.3 |

## Команды Telegram-бота

- `/start` — приветствие
- `/help` — справка

Любой текстовый вопрос обрабатывается через API и возвращает ответ из FAQ или сообщение о передаче администратору.
