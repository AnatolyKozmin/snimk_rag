"""API маршруты."""
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from services.normalizer import normalize_question

router = APIRouter()


@router.get("/", include_in_schema=False)
async def root():
    """Редирект на админ-панель."""
    return RedirectResponse(url="/admin", status_code=302)


@router.get("/health")
async def health(request: Request):
    """Проверка готовности сервиса."""
    ready = getattr(request.app.state, "ready", False)
    return {"status": "ready" if ready else "loading", "ready": ready}


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Пустой favicon, чтобы убрать 404 в консоли."""
    return Response(status_code=204)


class AskRequest(BaseModel):
    """Запрос на вопрос."""

    question: str
    telegram_user_id: Optional[int] = None


class AskResponse(BaseModel):
    """Ответ на вопрос."""

    answer: str
    confidence: float
    status: str  # "answered" | "pending"


def get_faq_service(request: Request):
    """Получить FAQ сервис из состояния приложения."""
    return request.app.state.faq_service


def get_pending_service(request: Request):
    """Получить Pending сервис."""
    return request.app.state.pending_service


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    body: AskRequest,
    request: Request,
):
    """
    Обработка вопроса пользователя.
    Возвращает ответ из FAQ или статус pending.
    """
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail="Сервис загружается, попробуйте через минуту.",
        )

    faq_service = request.app.state.faq_service
    pending_service = request.app.state.pending_service

    question = body.question.strip()
    if not question:
        return AskResponse(
            answer="Пожалуйста, задайте вопрос.",
            confidence=0.0,
            status="answered",
        )

    answer, confidence, status = await faq_service.search(question)

    if status == "answered" and answer:
        return AskResponse(answer=answer, confidence=confidence, status="answered")

    # Добавляем в очередь администратора
    normalized = normalize_question(question)
    await pending_service.add(
        question=question,
        question_normalized=normalized,
        telegram_user_id=body.telegram_user_id,
    )

    return AskResponse(
        answer="Ваш вопрос передан администратору. Мы ответим вам в ближайшее время.",
        confidence=confidence,
        status="pending",
    )
