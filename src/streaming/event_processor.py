"""
Processador de eventos em streaming (simulacao de tempo quase real).

Simula um pipeline Kafka-like usando queue.Queue + threading:
  - Producer gera eventos de usuarios com delay configuravel
  - Consumer processa cada evento: features -> modelo -> acao
  - Resultados sao acumulados e podem ser consultados

Uso:
  from src.streaming.event_processor import StreamProcessor, generate_synthetic_events

  processor = StreamProcessor(model_path="../models/rf_model.pkl")
  events = generate_synthetic_events(n=20)
  results = processor.run(events, delay=0.1)
"""

import queue
import threading
import time
import random
import joblib
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UserEvent:
    user_id: str
    days_since_last_purchase: float
    login_frequency: float
    session_duration_avg: float
    pages_per_session: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ProcessedEvent:
    user_id: str
    churn_score: float
    segment: str
    action: str
    processed_at: str
    latency_ms: float


def generate_synthetic_events(n: int = 50, seed: int = 42) -> List[UserEvent]:
    """Gera N eventos sinteticos de usuarios com perfis variados."""
    random.seed(seed)
    events = []
    profiles = [
        # (recency_range, freq_range, session_range, pages_range)
        ((1, 15),  (15, 30), (10, 20), (4, 8)),   # engajado
        ((20, 45), (5, 15),  (5, 12),  (2, 5)),   # ocasional
        ((60, 120),(1, 5),   (1, 6),   (1, 3)),   # risco
        ((90, 180),(0, 2),   (0, 3),   (0, 2)),   # inativo
    ]
    for i in range(n):
        p = random.choice(profiles)
        events.append(UserEvent(
            user_id=f"stream_u_{i:04d}",
            days_since_last_purchase=random.uniform(*p[0]),
            login_frequency=random.uniform(*p[1]),
            session_duration_avg=random.uniform(*p[2]),
            pages_per_session=random.uniform(*p[3]),
        ))
    return events


def _event_to_features(event: UserEvent) -> pd.DataFrame:
    return pd.DataFrame([{
        "recency_days":         event.days_since_last_purchase,
        "frequency":            event.login_frequency,
        "avg_session_duration": event.session_duration_avg,
        "intensity":            event.pages_per_session,
        "engagement_trend":     0.0,
    }])


def _score_to_segment(score: float) -> str:
    if score >= 0.7:
        return "at_risk"
    elif score >= 0.4:
        return "occasional"
    return "engaged"


def _score_to_action(segment: str, score: float) -> str:
    if score >= 0.85:
        return "priority_retention_call"
    return {
        "at_risk":   "offer_discount",
        "occasional":"engagement_campaign",
        "engaged":   "recommend_new_product",
    }.get(segment, "no_action")


class StreamProcessor:
    """
    Processador de eventos em streaming com produtor/consumidor.

    Usa duas threads:
      - Producer: empurra eventos para a fila com delay opcional
      - Consumer: consome e processa em tempo quase real
    """

    def __init__(self, model_path: str = "../models/rf_model.pkl"):
        self._model = joblib.load(model_path)
        self._queue: queue.Queue = queue.Queue()
        self._results: List[ProcessedEvent] = []
        self._lock = threading.Lock()

    def _process_event(self, event: UserEvent) -> ProcessedEvent:
        t0 = time.perf_counter()
        features = _event_to_features(event)
        if hasattr(self._model, "feature_names_in_"):
            features = features.reindex(
                columns=self._model.feature_names_in_, fill_value=0
            )
        score   = float(self._model.predict_proba(features)[0][1])
        segment = _score_to_segment(score)
        action  = _score_to_action(segment, score)
        latency = (time.perf_counter() - t0) * 1000

        return ProcessedEvent(
            user_id=event.user_id,
            churn_score=round(score, 4),
            segment=segment,
            action=action,
            processed_at=datetime.utcnow().isoformat(),
            latency_ms=round(latency, 3),
        )

    def _consumer(self, total: int, verbose: bool) -> None:
        processed = 0
        while processed < total:
            try:
                event = self._queue.get(timeout=5.0)
                result = self._process_event(event)
                with self._lock:
                    self._results.append(result)
                if verbose:
                    print(
                        f"[{result.processed_at[11:19]}] "
                        f"{result.user_id:16s} | "
                        f"score={result.churn_score:.3f} | "
                        f"seg={result.segment:12s} | "
                        f"acao={result.action:25s} | "
                        f"{result.latency_ms:.1f}ms"
                    )
                self._queue.task_done()
                processed += 1
            except queue.Empty:
                break

    def run(
        self,
        events: List[UserEvent],
        delay: float = 0.05,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Processa lista de eventos simulando stream.

        Args:
            events: lista de UserEvent
            delay: intervalo entre eventos em segundos (simula chegada real)
            verbose: imprime resultado de cada evento

        Returns:
            lista de dicionarios com resultados
        """
        self._results = []

        if verbose:
            print(f"Iniciando stream: {len(events)} eventos | delay={delay}s\n")
            print(f"{'Usuario':16s}  {'Score':>5}  {'Segmento':12s}  {'Acao':25s}  Latencia")
            print("-" * 80)

        consumer_thread = threading.Thread(
            target=self._consumer,
            args=(len(events), verbose),
            daemon=True,
        )
        consumer_thread.start()

        for event in events:
            self._queue.put(event)
            time.sleep(delay)

        consumer_thread.join(timeout=30)

        return [vars(r) for r in self._results]

    def summary(self) -> Dict[str, Any]:
        """Retorna estatisticas do stream processado."""
        if not self._results:
            return {}
        scores = [r.churn_score for r in self._results]
        latencies = [r.latency_ms for r in self._results]
        segments = {}
        for r in self._results:
            segments[r.segment] = segments.get(r.segment, 0) + 1
        return {
            "total_events":    len(self._results),
            "score_mean":      round(sum(scores) / len(scores), 4),
            "score_max":       round(max(scores), 4),
            "high_risk_count": sum(1 for s in scores if s >= 0.7),
            "latency_mean_ms": round(sum(latencies) / len(latencies), 2),
            "latency_max_ms":  round(max(latencies), 2),
            "segments":        segments,
        }
