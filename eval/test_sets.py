"""
Генерация тестовых наборов (срезов данных) для оценки RAG.
"""
import json
import random
from typing import List, Dict


def load_golden_set(path: str = "data/golden_set.json") -> List[Dict]:
    """Загружает эталонный набор."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["examples"]


def make_slice_by_length(examples: List[Dict], max_len: int = 100) -> List[Dict]:
    """Срез по длине вопроса (только короткие)."""
    return [e for e in examples if len(e["question"]) <= max_len]


def make_slice_random(examples: List[Dict], n: int = 3, seed: int = 42) -> List[Dict]:
    """Случайный срез из N примеров (воспроизводимо)."""
    random.seed(seed)
    return random.sample(examples, min(n, len(examples)))


def make_slice_by_topic(examples: List[Dict], keyword: str) -> List[Dict]:
    """Срез по теме: примеры, где вопрос или контекст содержат ключевое слово."""
    keyword_lower = keyword.lower()
    return [
        e for e in examples
        if keyword_lower in e["question"].lower() or keyword_lower in e["context"].lower()
    ]


def list_available_slices(examples: List[Dict]) -> Dict[str, int]:
    """Возвращает доступные срезы и их размеры."""
    return {
        "Все примеры": len(examples),
        "Короткие вопросы (≤100 символов)": len(make_slice_by_length(examples, 100)),
        "Случайная выборка (3)": len(make_slice_random(examples, 3)),
        "Тема 'Луна'": len(make_slice_by_topic(examples, "луна")),
        "Тема 'Париж'": len(make_slice_by_topic(examples, "париж")),
    }
