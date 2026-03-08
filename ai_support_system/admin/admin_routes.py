"""Маршруты админ-панели."""
import logging
from pathlib import Path
from typing import List

import numpy as np
from fastapi import APIRouter, File, Form, Request, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select

from database.models import PendingQuestion
from services.clustering_service import ClusteringService
from services.embedding_service import EmbeddingService
from services.learning_service import LearningService
from services.faq_loader import load_faq_from_bytes
from services.pending_service import PendingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# Шаблоны - путь относительно этого файла
_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))


class AnswerRequest(BaseModel):
    """Запрос на добавление ответа."""

    question: str
    answer: str


class ClusterAnswerRequest(BaseModel):
    """Запрос на ответ для кластера."""

    cluster_id: int
    questions: List[str]
    answer: str


@router.get("", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Главная страница админ-панели."""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "title": "FAQ Admin"},
    )


@router.get("/pending", response_class=JSONResponse)
async def get_pending_questions(request: Request):
    """
    Получить pending вопросы с кластеризацией.
    """
    pending_service = request.app.state.pending_service
    pending_list = await pending_service.get_pending(include_answered=False)

    if not pending_list:
        return JSONResponse(
            content={
                "clusters": [],
                "ungrouped": [],
                "total": 0,
            }
        )

    # Без кластеризации, если модель ещё не загружена
    if not getattr(request.app.state, "ready", False):
        ungrouped = [
            {
                "id": p.id,
                "question": p.question,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in pending_list
        ]
        return JSONResponse(
            content={"clusters": [], "ungrouped": ungrouped, "total": len(pending_list)}
        )

    embedding_service = request.app.state.embedding_service
    clustering_service = request.app.state.clustering_service
    questions = [p.question for p in pending_list]
    embeddings = embedding_service.embed_batch(questions)
    clusters_raw = clustering_service.cluster(embeddings, questions)

    # Преобразуем в формат для фронтенда
    clusters = []
    ungrouped = []
    pending_by_idx = {i: pending_list[i] for i in range(len(pending_list))}

    for cluster_id, items in clusters_raw:
        cluster_questions = [
            {
                "id": pending_by_idx[idx].id,
                "question": q,
                "created_at": pending_by_idx[idx].created_at.isoformat()
                if pending_by_idx[idx].created_at
                else None,
            }
            for idx, q in items
        ]
        if cluster_id >= 0 and len(items) > 1:
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "questions": cluster_questions,
                    "count": len(cluster_questions),
                }
            )
        else:
            for item in cluster_questions:
                ungrouped.append(item)

    return JSONResponse(
        content={
            "clusters": clusters,
            "ungrouped": ungrouped,
            "total": len(pending_list),
        }
    )


@router.post("/answer", response_class=JSONResponse)
async def submit_answer(
    request: Request,
    question: str = Form(...),
    answer: str = Form(...),
):
    """
    Отправить ответ на вопрос.
    Система автоматически добавит в FAQ и обновит FAISS.
    """
    learning_service = request.app.state.learning_service
    faq_service = request.app.state.faq_service

    if not question.strip() or not answer.strip():
        raise HTTPException(status_code=400, detail="Question and answer required")

    entry = await learning_service.add_qa(question.strip(), answer.strip())
    faq_service.invalidate_cache()

    return JSONResponse(
        content={
            "success": True,
            "faq_entry_id": entry.id,
            "message": "Ответ добавлен в FAQ",
        }
    )


@router.post("/answer/json", response_class=JSONResponse)
async def submit_answer_json(request: Request, body: AnswerRequest):
    """API endpoint для ответа (JSON)."""
    learning_service = request.app.state.learning_service
    faq_service = request.app.state.faq_service

    if not body.question.strip() or not body.answer.strip():
        raise HTTPException(status_code=400, detail="Question and answer required")

    entry = await learning_service.add_qa(body.question.strip(), body.answer.strip())
    faq_service.invalidate_cache()

    return JSONResponse(
        content={
            "success": True,
            "faq_entry_id": entry.id,
            "message": "Ответ добавлен в FAQ",
        }
    )


@router.post("/answer/pending/{pending_id}", response_class=JSONResponse)
async def submit_pending_answer(
    request: Request,
    pending_id: int,
    answer: str = Form(...),
):
    """Ответ на конкретный pending вопрос."""
    learning_service = request.app.state.learning_service
    faq_service = request.app.state.faq_service

    if not answer.strip():
        raise HTTPException(status_code=400, detail="Answer required")

    entry = await learning_service.add_qa_from_pending(pending_id, answer.strip())
    if not entry:
        raise HTTPException(status_code=404, detail="Pending question not found")

    faq_service.invalidate_cache()

    return JSONResponse(
        content={
            "success": True,
            "faq_entry_id": entry.id,
            "message": "Ответ добавлен в FAQ",
        }
    )


@router.post("/answer/cluster", response_class=JSONResponse)
async def submit_cluster_answer(request: Request, body: ClusterAnswerRequest):
    """
    Ответ на кластер похожих вопросов.
    Все вопросы кластера получат один ответ.
    """
    learning_service = request.app.state.learning_service
    faq_service = request.app.state.faq_service
    session_factory = request.app.state.session_factory

    if not body.questions or not body.answer.strip():
        raise HTTPException(status_code=400, detail="Questions and answer required")

    entry = await learning_service.add_qa_for_cluster(body.questions, body.answer.strip())

    # Пометить все pending вопросы из кластера как отвеченные
    async with session_factory() as session:
        for q in body.questions:
            result = await session.execute(
                select(PendingQuestion).where(
                    PendingQuestion.question == q,
                    PendingQuestion.answered == False,
                )
            )
            for pending in result.scalars().all():
                pending.answered = True
                pending.faq_entry_id = entry.id
        await session.commit()

    faq_service.invalidate_cache()

    return JSONResponse(
        content={
            "success": True,
            "faq_entry_id": entry.id,
            "message": "Ответ добавлен для всего кластера",
        }
    )


@router.get("/export", response_class=JSONResponse)
async def export_faq(request: Request):
    """Экспорт FAQ базы в JSON."""
    faq_service = request.app.state.faq_service
    entries = await faq_service.get_all_entries()
    data = [
        {
            "id": e.id,
            "question": e.question,
            "answer": e.answer,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in entries
    ]
    return JSONResponse(content={"faq": data, "total": len(data)})


@router.post("/import", response_class=JSONResponse)
async def import_faq(
    request: Request,
    file: UploadFile = File(...),
    clear: str = Form(""),
):
    """
    Импорт FAQ из Excel (.xlsx) или CSV.
    Формат: колонка 1 = вопрос, колонка 2 = ответ.
    clear=true — очистить базу перед импортом.
    """
    from database.models import FAQEntry
    from sqlalchemy import delete

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только .csv и .xlsx файлы",
        )

    content = await file.read()
    try:
        pairs = load_faq_from_bytes(content, file.filename or "upload")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {e}")

    if not pairs:
        raise HTTPException(
            status_code=400,
            detail="Нет данных. Проверьте формат: колонка 1 = вопрос, колонка 2 = ответ.",
        )

    do_clear = str(clear).lower() in ("true", "on", "1", "yes")
    learning_service = request.app.state.learning_service
    faq_service = request.app.state.faq_service
    session_factory = request.app.state.session_factory
    faiss_index = request.app.state.faiss_index

    if do_clear:
        async with session_factory() as session:
            await session.execute(delete(FAQEntry))
            await session.commit()
        dim = request.app.state.embedding_service.dimension
        faiss_index.rebuild(np.array([]).reshape(0, dim), [])
        faiss_index.save()

    added = 0
    for question, answer in pairs:
        await learning_service.add_qa(question, answer)
        added += 1

    faq_service.invalidate_cache()

    return JSONResponse(
        content={
            "success": True,
            "message": f"Импортировано {added} записей",
            "count": added,
        }
    )


@router.post("/rebuild-index", response_class=JSONResponse)
async def rebuild_index(request: Request):
    """Пересборка FAISS индекса из базы данных."""
    from database.models import FAQEntry

    session_factory = request.app.state.session_factory
    embedding_service = request.app.state.embedding_service
    faiss_index = request.app.state.faiss_index
    faq_service = request.app.state.faq_service

    async with session_factory() as session:
        result = await session.execute(select(FAQEntry))
        entries = list(result.scalars().all())

    if not entries:
        return JSONResponse(
            content={"success": True, "message": "No entries to rebuild", "count": 0}
        )

    questions = [e.question for e in entries]
    ids = [e.id for e in entries]
    embeddings = embedding_service.embed_batch(questions)
    faiss_index.rebuild(embeddings, ids)
    faiss_index.save()
    faq_service.invalidate_cache()

    return JSONResponse(
        content={
            "success": True,
            "message": "FAISS index rebuilt",
            "count": len(entries),
        }
    )
