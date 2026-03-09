"""Telegram бот на aiogram."""
import asyncio
import logging
from typing import Optional

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

logger = logging.getLogger(__name__)

# Повторы при 503 (сервис загружается): паузы в секундах
RETRY_DELAYS = [15, 30, 45]


async def get_answer_from_api(
    api_url: str,
    question: str,
    telegram_user_id: Optional[int] = None,
    retry_on_503: bool = True,
) -> dict:
    """Запрос к API бэкенда. При 503 повторяет с паузами (15, 30, 45 сек)."""
    payload = {"question": question}
    if telegram_user_id is not None:
        payload["telegram_user_id"] = telegram_user_id

    for attempt in range(len(RETRY_DELAYS) + 1):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/ask",
                json=payload,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code == 503 and retry_on_503 and attempt < len(RETRY_DELAYS):
                delay = RETRY_DELAYS[attempt]
                logger.info("API 503, повтор через %d сек (попытка %d)", delay, attempt + 1)
                await asyncio.sleep(delay)
                continue
            response.raise_for_status()

    raise httpx.HTTPStatusError("Service unavailable", request=response.request, response=response)


def create_bot(token: str, api_url: str) -> tuple[Bot, Dispatcher]:
    """Создать бота и диспетчер."""

    bot = Bot(token=token)
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def cmd_start(message: Message):
        await message.answer(
            "Привет! Я FAQ-ассистент. Задайте мне любой вопрос, "
            "и я постараюсь найти ответ в базе знаний.\n\n"
            "Команды:\n"
            "/help - справка"
        )

    @dp.message(Command("help"))
    async def cmd_help(message: Message):
        await message.answer(
            "Просто напишите свой вопрос текстом, и я поищу ответ в FAQ.\n\n"
            "Если ответ не найден, ваш вопрос будет передан администратору."
        )

    @dp.message(F.text)
    async def handle_question(message: Message):
        question = message.text.strip()
        if not question:
            await message.answer("Пожалуйста, задайте вопрос.")
            return

        # Показываем "печатает" пока ждём ответ
        await message.bot.send_chat_action(message.chat.id, "typing")

        try:
            user_id = message.from_user.id if message.from_user else None
            data = await get_answer_from_api(api_url, question, user_id)
            answer = data.get("answer", "Произошла ошибка. Попробуйте позже.")
            status = data.get("status", "pending")

            await message.answer(answer)

            if status == "pending":
                logger.info(
                    "Question escalated to admin: user_id=%s, question=%s",
                    message.from_user.id if message.from_user else None,
                    question[:50],
                )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                await message.answer(
                    "Сервис ещё загружается (модель ~2 мин). Попробуйте через минуту."
                )
            else:
                logger.exception("API request failed: %s", e)
                await message.answer("Временная ошибка сервиса. Попробуйте позже.")
        except httpx.HTTPError as e:
            logger.exception("API request failed: %s", e)
            await message.answer(
                "Временная ошибка сервиса. Пожалуйста, попробуйте позже."
            )
        except Exception as e:
            logger.exception("Unexpected error: %s", e)
            await message.answer("Произошла ошибка. Попробуйте позже.")

    return bot, dp


async def run_bot(token: str, api_url: str) -> None:
    """Запуск бота."""
    bot, dp = create_bot(token, api_url)
    await dp.start_polling(bot)
