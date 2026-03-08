"""Загрузка FAQ из Excel и CSV."""
import csv
from pathlib import Path
from typing import List, Tuple


def _is_header_row(q: str, a: str) -> bool:
    """Проверка, похожа ли строка на заголовок."""
    ql, al = q.lower(), a.lower()
    return (ql in ("вопрос", "question", "q") and al in ("ответ", "answer", "a")) or (
        ql == "question" and al == "answer"
    )


def load_from_csv(content: bytes, filename: str = "") -> List[Tuple[str, str]]:
    """Загрузить Q/A из CSV (UTF-8). Пробует запятую и точку с запятой."""
    text = content.decode("utf-8-sig")
    rows: List[Tuple[str, str]] = []
    for delim in (",", ";"):
        reader = csv.reader(text.splitlines(), delimiter=delim)
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                q, a = row[0].strip(), row[1].strip()
                if not _is_header_row(q, a):
                    rows.append((q, a))
        if rows:
            break
    return rows


def load_from_excel(content: bytes, filename: str = "") -> List[Tuple[str, str]]:
    """Загрузить Q/A из Excel. Первый лист, колонки A (вопрос) и B (ответ)."""
    import io

    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    ws = wb.active
    rows: List[Tuple[str, str]] = []
    for row in ws.iter_rows(min_row=1, values_only=True):
        if not row or len(row) < 2:
            continue
        q, a = str(row[0] or "").strip(), str(row[1] or "").strip()
        if not q or not a:
            continue
        if _is_header_row(q, a):
            continue
        rows.append((q, a))
    wb.close()
    return rows


def load_faq_from_bytes(content: bytes, filename: str = "upload") -> List[Tuple[str, str]]:
    """Загрузить FAQ из содержимого файла по расширению."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return load_from_csv(content, filename)
    if suffix in (".xlsx", ".xls"):
        return load_from_excel(content, filename)
    raise ValueError(f"Неизвестный формат: {suffix}. Поддерживаются .csv, .xlsx")
