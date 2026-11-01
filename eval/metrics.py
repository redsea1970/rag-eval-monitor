"""
Модуль метрик для оценки RAG-системы.
Все метрики работают в демо-режиме (без API-ключей).
"""
import re
from typing import List, Dict


def _normalize(text: str) -> str:
    """Приводит текст к нижнему регистру и удаляет лишние символы."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return " ".join(text.split())


def _tokenize(text: str) -> List[str]:
    """Разбивает текст на токены (слова)."""
    return _normalize(text).split()


def context_precision(retrieved_chunks: List[str], relevant_chunks: List[str]) -> float:
    """Точность контекста: какой процент найденных фрагментов релевантен."""
    if not retrieved_chunks:
        return 0.0
    relevant_set = {_normalize(c) for c in relevant_chunks}
    hits = sum(1 for c in retrieved_chunks if _normalize(c) in relevant_set)
    return hits / len(retrieved_chunks)


def context_recall(retrieved_chunks: List[str], relevant_chunks: List[str]) -> float:
    """Полнота контекста: какой процент релевантных фрагментов найден."""
    if not relevant_chunks:
        return 0.0
    retrieved_set = {_normalize(c) for c in retrieved_chunks}
    hits = sum(1 for c in relevant_chunks if _normalize(c) in retrieved_set)
    return hits / len(relevant_chunks)


def exact_match(generated_answer: str, reference_answer: str) -> float:
    """Точное совпадение: 1.0 если ответы совпадают, иначе 0.0."""
    return 1.0 if _normalize(generated_answer) == _normalize(reference_answer) else 0.0


def token_overlap_f1(generated_answer: str, reference_answer: str) -> float:
    """Лексическое сходство (F1 по токенам)."""
    gen_tokens = set(_tokenize(generated_answer))
    ref_tokens = set(_tokenize(reference_answer))
    if not gen_tokens or not ref_tokens:
        return 0.0
    common = gen_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(gen_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def factuality_score(generated_answer: str, context: str) -> float:
    """Грубая оценка согласованности фактов: доля токенов ответа, подтверждённых контекстом."""
    answer_tokens = set(_tokenize(generated_answer))
    context_tokens = set(_tokenize(context))
    if not answer_tokens:
        return 0.0
    supported = answer_tokens & context_tokens
    return len(supported) / len(answer_tokens)


def evaluate_example(example: Dict, retrieved_chunks: List[str], generated_answer: str) -> Dict:
    """Оценивает один пример по всем метрикам."""
    return {
        "id": example["id"],
        "context_precision": context_precision(retrieved_chunks, example["relevant_chunks"]),
        "context_recall": context_recall(retrieved_chunks, example["relevant_chunks"]),
        "exact_match": exact_match(generated_answer, example["reference_answer"]),
        "token_overlap_f1": token_overlap_f1(generated_answer, example["reference_answer"]),
        "factuality_score": factuality_score(generated_answer, example["context"]),
    }


def aggregate_metrics(results: List[Dict]) -> Dict:
    """Усредняет метрики по всем примерам."""
    if not results:
        return {}
    keys = ["context_precision", "context_recall", "exact_match", "token_overlap_f1", "factuality_score"]
    return {k: sum(r[k] for r in results) / len(results) for k in keys}
