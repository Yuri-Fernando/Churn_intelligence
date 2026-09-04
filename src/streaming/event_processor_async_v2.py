"""
src/streaming/event_processor_async_v2.py

V2 do processador de eventos em streaming: a V1 (event_processor.py) usa
uma única thread consumidora sobre uma `queue.Queue`. Esta V2 reimplementa
o mesmo contrato (mesmos `UserEvent`/`ProcessedEvent`, mesma lógica de
score -> segmento -> ação) usando `asyncio`, com múltiplos workers
concorrentes consumindo de uma `asyncio.Queue` -- útil quando o "processar
evento" passa a envolver I/O real (chamada a um modelo servido via API,
gravação em um banco de eventos, etc.) e não apenas CPU local.

Uso:
    import asyncio
    from src.streaming.event_processor_async_v2 import AsyncStreamProcessor
    from src.streaming.event_processor import generate_synthetic_events

    processor = AsyncStreamProcessor(model_path="../models/rf_model.pkl", n_workers=4)
    events = generate_synthetic_events(n=50)
    results = asyncio.run(processor.run(events, delay=0.01))
"""

import asyncio
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List

import joblib

from src.streaming.event_processor import (
    ProcessedEvent,
    UserEvent,
    _event_to_features,
    _score_to_action,
    _score_to_segment,
)


class AsyncStreamProcessor:
    """
    Processador de eventos assíncrono com N workers concorrentes.

    Arquitetura:
      - Um producer assíncrono empurra eventos para uma `asyncio.Queue`
        (com delay configurável, simulando chegada em tempo real).
      - `n_workers` corrotinas consomem da mesma fila concorrentemente --
        cada uma roda a predição (bloqueante) em uma thread separada via
        `asyncio.to_thread`, para não travar o loop de eventos enquanto o
        modelo calcula.
    """

    def __init__(self, model_path: str = "../models/rf_model.pkl", n_workers: int = 4):
        self._model = joblib.load(model_path)
        self._n_workers = n_workers
        self._results: List[ProcessedEvent] = []
        self._lock = asyncio.Lock()

    def _predict(self, event: UserEvent) -> ProcessedEvent:
        """Parte síncrona/bloqueante (feature building + inferência do
        modelo) -- roda em thread separada para não bloquear o event loop."""
        t0 = time.perf_counter()
        features = _event_to_features(event)
        if hasattr(self._model, "feature_names_in_"):
            features = features.reindex(columns=self._model.feature_names_in_, fill_value=0)

        score = float(self._model.predict_proba(features)[0][1])
        segment = _score_to_segment(score)
        action = _score_to_action(segment, score)
        latency = (time.perf_counter() - t0) * 1000

        return ProcessedEvent(
            user_id=event.user_id,
            churn_score=round(score, 4),
            segment=segment,
            action=action,
            processed_at=datetime.utcnow().isoformat(),
            latency_ms=round(latency, 3),
        )

    async def _worker(self, worker_id: int, queue: "asyncio.Queue[UserEvent]", verbose: bool) -> None:
        while True:
            event = await queue.get()
            if event is None:  # sentinela de encerramento
                queue.task_done()
                break

            result = await asyncio.to_thread(self._predict, event)

            async with self._lock:
                self._results.append(result)

            if verbose:
                print(
                    f"[worker {worker_id}] [{result.processed_at[11:19]}] "
                    f"{result.user_id:16s} | score={result.churn_score:.3f} | "
                    f"seg={result.segment:12s} | acao={result.action:25s} | "
                    f"{result.latency_ms:.1f}ms"
                )

            queue.task_done()

    async def run(
        self, events: List[UserEvent], delay: float = 0.01, verbose: bool = True
    ) -> List[Dict[str, Any]]:
        self._results = []
        queue: "asyncio.Queue[UserEvent]" = asyncio.Queue()

        if verbose:
            print(
                f"Iniciando stream assincrono: {len(events)} eventos "
                f"| {self._n_workers} workers | delay={delay}s\n"
            )

        workers = [
            asyncio.create_task(self._worker(i, queue, verbose)) for i in range(self._n_workers)
        ]

        for event in events:
            await queue.put(event)
            await asyncio.sleep(delay)

        for _ in workers:
            await queue.put(None)  # um sentinela por worker

        await queue.join()
        await asyncio.gather(*workers)

        return [asdict(r) for r in self._results]

    def summary(self) -> Dict[str, Any]:
        if not self._results:
            return {}
        scores = [r.churn_score for r in self._results]
        latencies = [r.latency_ms for r in self._results]
        segments: Dict[str, int] = {}
        for r in self._results:
            segments[r.segment] = segments.get(r.segment, 0) + 1
        return {
            "total_events": len(self._results),
            "workers": self._n_workers,
            "score_mean": round(sum(scores) / len(scores), 4),
            "score_max": round(max(scores), 4),
            "high_risk_count": sum(1 for s in scores if s >= 0.7),
            "latency_mean_ms": round(sum(latencies) / len(latencies), 2),
            "latency_max_ms": round(max(latencies), 2),
            "segments": segments,
        }
