"""Telegram бот на aiogram."""
import logging
from typing import Optional

import httpx
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import Message

logger = logging.getLogger(__name__)

ESCALATION_MESSAGE = (
    "Ваш вопрос передан администратору. "
    "Мы ответим вам в ближайшее время."
)


async def get_answer_from_api(
    api_url: str, question: str, telegram_user_id: Optional[int] = None
) -> dict:
    """Запрос к API бэкенда."""
    payload = {"question": question}
    if telegram_user_id is not None:
        payload["telegram_user_id"] = telegram_user_id
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{api_url}/ask",
            json=payload,
        )
        response.raise_for_status()
        return response.json()


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
