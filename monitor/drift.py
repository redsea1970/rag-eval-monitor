"""
Обнаружение дрейфа (drift detection) в RAG-системе.
Сравнивает текущее распределение метрик с эталонным.
"""
from typing import List, Dict
import statistics


def length_distribution(records: List[Dict], field: str) -> List[int]:
    """Распределение длин (в символах) по указанному полю."""
    return [len(r.get(field, "")) for r in records]


def detect_length_drift(
    baseline: List[int],
    current: List[int],
    threshold: float = 0.3,
) -> Dict:
    """
    Обнаруживает дрейф в длине ответов.
    Если относительное изменение среднего превышает threshold — сигнал о дрейфе.
    """
    if not baseline or not current:
        return {"drift": False, "reason": "недостаточно данных"}
    base_mean = statistics.mean(baseline)
    curr_mean = statistics.mean(current)
    if base_mean == 0:
        return {"drift": False, "reason": "нулевая база"}
    change = abs(curr_mean - base_mean) / base_mean
    return {
        "drift": change > threshold,
        "baseline_mean": round(base_mean, 1),
        "current_mean": round(curr_mean, 1),
        "relative_change": round(change, 3),
        "threshold": threshold,
    }


def detect_metric_drift(
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    threshold: float = 0.1,
) -> Dict:
    """
    Сравнивает метрики между baseline и текущими.
    Если относительное изменение больше threshold — сигнал о дрейфе.
    """
    drifts = {}
    for key, base_val in baseline_metrics.items():
        if key in current_metrics:
            curr_val = current_metrics[key]
            if base_val > 0:
                change = (curr_val - base_val) / base_val
                drifts[key] = {
                    "baseline": round(base_val, 3),
                    "current": round(curr_val, 3),
                    "relative_change": round(change, 3),
                    "drift": abs(change) > threshold,
                }
    any_drift = any(v["drift"] for v in drifts.values())
    return {"any_drift": any_drift, "details": drifts}
