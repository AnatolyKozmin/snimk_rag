"""Сервис LLM для RAG-ответов на базе Qwen2.5-1.5B-Instruct."""
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Размерность по умолчанию (для совместимости)
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

RAG_SYSTEM_PROMPT = """Ты помощник службы поддержки. Отвечай кратко и по делу на русском языке.
Используй ТОЛЬКО информацию из контекста FAQ ниже. Не выдумывай.
Если в контексте нет ответа на вопрос — напиши: "К сожалению, в базе знаний нет ответа на этот вопрос. Обратитесь к администратору."
Отвечай одним абзацем, без нумерации и лишних вступлений."""

RAG_USER_TEMPLATE = """Контекст из FAQ:
{context}

Вопрос пользователя: {question}

Ответ:"""


class LLMService:
    """Сервис LLM для генерации RAG-ответов. Ленивая загрузка модели."""

    _instance: Optional["LLMService"] = None

    def __new__(cls, model_name: str = DEFAULT_LLM_MODEL, cache_dir: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL, cache_dir: Optional[str] = None):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Загрузить модель при первом обращении."""
        if self._model is not None:
            return

        logger.info("Loading LLM: %s", self._model_name)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {}
        if self._cache_dir:
            kwargs["cache_dir"] = self._cache_dir

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True, **kwargs)
        # float16 на GPU, float32 на CPU (float16 на CPU может быть нестабилен)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        if not torch.cuda.is_available():
            self._model = self._model.to("cpu")

        logger.info("LLM loaded")

    def generate_rag_answer(
        self,
        question: str,
        faq_entries: List[Tuple[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ) -> str:
        """
        Сгенерировать ответ на вопрос на основе контекста FAQ.
        faq_entries: список (question, answer) из FAQ.
        """
        self._ensure_loaded()

        if not faq_entries:
            return "К сожалению, в базе знаний нет ответа на этот вопрос. Обратитесь к администратору."

        context_parts = []
        for i, (q, a) in enumerate(faq_entries, 1):
            context_parts.append(f"{i}. В: {q}\n   О: {a}")
        context = "\n\n".join(context_parts)

        user_text = RAG_USER_TEMPLATE.format(context=context, question=question)

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        import torch

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        output_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
        response = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response.strip()

    @property
    def is_loaded(self) -> bool:
        """Модель загружена."""
        return self._model is not None
