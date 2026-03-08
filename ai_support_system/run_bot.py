#!/usr/bin/env python3
"""
Точка входа для Telegram-бота (отдельный сервис).
Запуск: python run_bot.py
В Docker: API_URL=http://faq_api:8000
"""
import asyncio
import logging
import sys

from core.config import settings
from bot.telegram_bot import run_bot

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не задан. Выйход.")
        sys.exit(1)

    api_url = settings.API_URL.rstrip("/")
    logger.info("Bot starting, API: %s", api_url)
    asyncio.run(run_bot(settings.TELEGRAM_BOT_TOKEN, api_url))
