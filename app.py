"""
RAG-Eval-Monitor: Streamlit-дашборд для оценки и мониторинга RAG-систем.
Всё в демо-режиме, без API-ключей.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px

from eval.metrics import evaluate_example, aggregate_metrics
from eval.judge import TFIDFJudge, interpret_score
from eval.test_sets import (
    load_golden_set,
    list_available_slices,
    make_slice_by_length,
    make_slice_by_topic,
)
from monitor.logger import log_request, load_logs, clear_logs
from monitor.drift import detect_length_drift, detect_metric_drift


st.set_page_config(page_title="RAG-Eval-Monitor", page_icon="🔍", layout="wide")
st.title("🔍 RAG-Eval-Monitor")
st.caption("Конвейер оценки и мониторинга RAG-систем (демо-режим, без API)")


with st.sidebar:
    st.header("Настройки")
    demo_answer_mode = st.selectbox(
        "Режим ответа:",
        ["Идеальный", "Случайный", "Пустой"],
        help="Имитация ответа RAG-системы для демонстрации метрик",
    )
    st.markdown("---")
    if st.button("Очистить логи мониторинга"):
        clear_logs()
        st.success("Логи очищены")

examples = load_golden_set()
judge = TFIDFJudge()


def simulate_answer(example, mode: str) -> str:
    """Имитация ответа RAG-системы."""
    if mode == "Идеальный":
        return example["reference_answer"]
    if mode == "Случайный":
        return example["context"][:50]
    return ""


tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Оценка", "🧑‍⚖️ AI-судья", "📈 Мониторинг", "🔀 Дрейф"]
)


# ========== ВКЛАДКА 1: ОЦЕНКА ==========
with tab1:
    st.subheader("Оценка RAG-системы на эталонном наборе")
    st.markdown(f"**Загружено примеров:** {len(examples)}")

    slices = list_available_slices(examples)
    st.markdown("**Доступные срезы данных:**")
    st.dataframe(
        pd.DataFrame([{"Срез": k, "Размер": v} for k, v in slices.items()]),
        use_container_width=True,
    )

    selected_slice = st.selectbox(
        "Выбери срез для оценки:",
        ["Все примеры", "Короткие вопросы (≤100 символов)", "Тема 'Луна'", "Тема 'Париж'"],
    )

    if selected_slice == "Все примеры":
        subset = examples
    elif selected_slice.startswith("Короткие"):
        subset = make_slice_by_length(examples, 100)
    elif "Луна" in selected_slice:
        subset = make_slice_by_topic(examples, "лун")
    elif "Париж" in selected_slice:
        subset = make_slice_by_topic(examples, "париж")
    else:
        subset = examples

    if st.button("▶️ Запустить оценку", type="primary"):
        results = []
        for ex in subset:
            answer = simulate_answer(ex, demo_answer_mode)
            start = time.time()
            metrics = evaluate_example(ex, [ex["context"]], answer)
            latency = (time.time() - start) * 1000
            results.append(metrics)
            log_request(ex["question"], [ex["context"]], answer, latency)

        df = pd.DataFrame(results)
        agg = aggregate_metrics(results)

        st.markdown("### Средние метрики")
        cols = st.columns(5)
        cols[0].metric("Точность контекста", f"{agg['context_precision']:.2f}")
        cols[1].metric("Полнота контекста", f"{agg['context_recall']:.2f}")
        cols[2].metric("Exact Match", f"{agg['exact_match']:.2f}")
        cols[3].metric("Token F1", f"{agg['token_overlap_f1']:.2f}")
        cols[4].metric("Согласованность фактов", f"{agg['factuality_score']:.2f}")

        st.markdown("### Детали по примерам")
        st.dataframe(df, use_container_width=True)

        fig = px.bar(
            df.melt(id_vars="id", var_name="Метрика", value_name="Значение"),
            x="id", y="Значение", color="Метрика", barmode="group",
            title="Метрики по примерам",
        )
        st.plotly_chart(fig, use_container_width=True)


# ========== ВКЛАДКА 2: AI-СУДЬЯ ==========
with tab2:
    st.subheader("AI-судья на основе TF-IDF (демо-версия)")
    st.caption(
        "Оценивает релевантность и обоснованность ответа. "
        "В реальном проекте здесь была бы LLM."
    )

    ex_choice = st.selectbox(
        "Выбери пример:",
        examples,
        format_func=lambda x: f"{x['id']}: {x['question']}",
    )
    generated = st.text_area(
        "Сгенерированный ответ (можно отредактировать):",
        value=ex_choice["reference_answer"],
    )

    if st.button("🧑‍⚖️ Оценить ответ"):
        scores = judge.evaluate(ex_choice, generated)
        col1, col2 = st.columns(2)
        col1.metric(
            "Релевантность",
            f"{scores['judge_relevance']:.2f}",
            interpret_score(scores["judge_relevance"]),
        )
        col2.metric(
            "Обоснованность",
            f"{scores['judge_groundedness']:.2f}",
            interpret_score(scores["judge_groundedness"]),
        )
        st.markdown("**Эталонный ответ:** " + ex_choice["reference_answer"])
        st.markdown("**Контекст:** " + ex_choice["context"])


# ========== ВКЛАДКА 3: МОНИТОРИНГ ==========
with tab3:
    st.subheader("Мониторинг запросов")
    logs = load_logs()
    st.markdown(f"**Всего записей:** {len(logs)}")

    if logs:
        df_logs = pd.DataFrame(logs)
        st.dataframe(df_logs, use_container_width=True)

        if "latency_ms" in df_logs.columns:
            fig = px.histogram(
                df_logs, x="latency_ms", nbins=20,
                title="Распределение времени отклика (мс)",
            )
            st.plotly_chart(fig, use_container_width=True)

        df_logs["answer_len"] = df_logs["generated_answer"].apply(len)
        fig2 = px.histogram(
            df_logs, x="answer_len", nbins=20,
            title="Распределение длины ответа (символы)",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Логов пока нет. Запусти оценку на вкладке «Оценка».")


# ========== ВКЛАДКА 4: ДРЕЙФ ==========
with tab4:
    st.subheader("Обнаружение дрейфа")
    st.caption(
        "Сравнивает эталонные метрики с текущими. "
        "В реальности здесь были бы данные из production."
    )

    logs = load_logs()
    if len(logs) < 6:
        st.warning("Нужно минимум 6 записей в логах для обнаружения дрейфа.")
    else:
        df_logs = pd.DataFrame(logs)
        df_logs["answer_len"] = df_logs["generated_answer"].apply(len)
        half = len(df_logs) // 2
        baseline = df_logs.iloc[:half]["answer_len"].tolist()
        current = df_logs.iloc[half:]["answer_len"].tolist()

        result = detect_length_drift(baseline, current, threshold=0.3)
        st.markdown("### Дрейф длины ответа")
        st.json(result)

        if result.get("drift"):
            st.error("⚠️ Обнаружен дрейф в длине ответов!")
        else:
            st.success("✅ Дрейфа не обнаружено.")

        st.markdown("---")
        st.markdown("### Сравнение метрик baseline vs current")
        baseline_metrics = {
            "context_precision": 0.85,
            "context_recall": 0.80,
            "factuality_score": 0.90,
        }
        current_metrics = {
            "context_precision": 0.60,
            "context_recall": 0.75,
            "factuality_score": 0.55,
        }
        drift_metrics = detect_metric_drift(
            baseline_metrics, current_metrics, threshold=0.1
        )
        st.json(drift_metrics)
