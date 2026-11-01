"""
Модуль логирования запросов и ответов для мониторинга.
Пишет всё в JSONL-файл (одна JSON-строка на запрос).
"""
import json
import os
from datetime import datetime
from typing import Dict, List


LOG_PATH = "monitor/logs.jsonl"


def log_request(
    question: str,
    retrieved_chunks: List[str],
    generated_answer: str,
    latency_ms: float = 0.0,
    metadata: Dict = None,
    log_path: str = LOG_PATH,
) -> None:
    """Логирует один запрос."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "generated_answer": generated_answer,
        "latency_ms": latency_ms,
        "metadata": metadata or {},
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_logs(log_path: str = LOG_PATH) -> List[Dict]:
    """Загружает все логи."""
    if not os.path.exists(log_path):
        return []
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clear_logs(log_path: str = LOG_PATH) -> None:
    """Очищает логи (для демонстрации)."""
    if os.path.exists(log_path):
        os.remove(log_path)
