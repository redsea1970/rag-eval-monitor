"""
AI-судья в демо-режиме (без API-ключей).
Использует TF-IDF и косинусное сходство вместо LLM.
"""
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFJudge:
    """Судья на основе TF-IDF."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
        )

    def score_relevance(self, generated_answer: str, reference_answer: str) -> float:
        """Релевантность: косинусное сходство между ответами (0..1)."""
        try:
            tfidf = self.vectorizer.fit_transform([generated_answer, reference_answer])
            return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except ValueError:
            return 0.0

    def score_groundedness(self, generated_answer: str, context: str) -> float:
        """Обоснованность: насколько ответ подтверждается контекстом (0..1)."""
        try:
            tfidf = self.vectorizer.fit_transform([generated_answer, context])
            return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except ValueError:
            return 0.0

    def evaluate(self, example: Dict, generated_answer: str) -> Dict:
        """Оценивает один пример."""
        return {
            "id": example["id"],
            "judge_relevance": self.score_relevance(generated_answer, example["reference_answer"]),
            "judge_groundedness": self.score_groundedness(generated_answer, example["context"]),
        }


def interpret_score(score: float) -> str:
    """Человекочитаемая интерпретация оценки."""
    if score >= 0.75:
        return "хорошо"
    if score >= 0.5:
        return "средне"
    if score >= 0.25:
        return "слабо"
    return "плохо"
