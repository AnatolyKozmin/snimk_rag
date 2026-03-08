"""Единая настройка логирования для всех сервисов."""
import logging
import sys

from .config import settings


def setup_logging(service_name: str = "") -> None:
    """
    Настройка логирования.
    service_name — префикс для Docker (api, bot), чтобы различать логи.
    """
    prefix = f"[{service_name}] " if service_name else ""
    fmt = f"%(asctime)s | %(levelname)-5s | {prefix}%(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Уменьшить шум от сторонних библиотек
    for name in (
        "httpx",
        "httpcore",
        "sentence_transformers",
        "huggingface_hub",
        "transformers",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Uvicorn access — логи каждого HTTP-запроса (очень шумно)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
