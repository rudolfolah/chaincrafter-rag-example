"""Persist completed local evaluations for the native TruLens dashboard."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from threading import get_native_id
from typing import TYPE_CHECKING
from uuid import uuid4


if TYPE_CHECKING:
    from eval_ollama import EvalResult


def _retrieved_chunks(query):
    if query.retrieved_chunks:
        return [
            {
                "rank": rank,
                "text": chunk.text,
                "original_text": chunk.original_text,
                "document_index": chunk.document_index,
                "source": str(chunk.source) if chunk.source is not None else None,
                "score": chunk.score,
            }
            for rank, chunk in enumerate(query.retrieved_chunks, start=1)
        ]
    return [
        {"rank": rank, "original_text": snippet}
        for rank, snippet in enumerate(query.original_snippets, start=1)
    ]


def save_results(
    database: Path,
    results: Sequence["EvalResult"],
    relevance_scores: Sequence[tuple[float, ...]],
    *,
    judge_model: str,
) -> None:
    """Append scored results and captured retrieval spans without rerunning models.

    App names group each demo; versions describe the variant and retrieval
    configuration. Record tags and metadata identify each invocation separately.
    Alignment is checked before opening the database so a missing score cannot
    leave a partially written evaluation.
    """
    if len(results) != len(relevance_scores):
        raise ValueError("Each evaluation result must have relevance scores.")
    for result, scores in zip(results, relevance_scores, strict=True):
        if len(result.queries) != len(scores):
            raise ValueError(
                f"{result.demo}/{result.variant}: each query must have a relevance score."
            )
    if not results:
        return

    # TruLens' legacy record APIs check this variable at call time. Keep imports,
    # session initialization and writes inside the scope, restoring the caller's
    # setting even when initialization or persistence fails.
    previous_tracing = os.environ.get("TRULENS_OTEL_TRACING")
    os.environ["TRULENS_OTEL_TRACING"] = "0"
    try:
        from sqlalchemy import URL, create_engine
        from trulens.core import TruSession
        from trulens.core.schema.app import AppDefinition
        from trulens.core.schema.base import Cost, Perf
        from trulens.core.schema.feedback import (
            FeedbackCall,
            FeedbackDefinition,
            FeedbackMode,
            FeedbackResult,
            FeedbackResultStatus,
        )
        from trulens.core.schema.record import (
            Record,
            RecordAppCall,
            RecordAppCallMethod,
        )
        from trulens.core.utils.pyschema import Class, Method, Obj
        from trulens.core.utils.serial import Lens
        from retriever import Retriever

        database = database.expanduser().resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        # Pass the URL object straight to SQLAlchemy: a string round-trip would
        # reinterpret literal '?' characters in a SQLite filename as URL queries.
        engine = create_engine(URL.create("sqlite", database=str(database)))
        run_id = uuid4().hex
        run_started_at = datetime.now(timezone.utc).isoformat()
        session_name = f"ollama-eval-writer-{run_id}"
        try:
            # A private named session cannot accidentally reuse another caller's
            # singleton and write to its database instead of the requested path.
            session = TruSession(name=session_name, database_engine=engine)
            feedbacks = (
                FeedbackDefinition(
                    feedback_definition_id="ollama_eval_context_relevance_v1",
                    name="Context Relevance",
                    higher_is_better=True,
                    description=(
                        "Mean of the existing TruLens built-in context relevance "
                        "scores for the retrieved original snippets; no judge rerun."
                    ),
                ),
                FeedbackDefinition(
                    feedback_definition_id="ollama_eval_hit_at_k_v1",
                    name="Hit@k",
                    higher_is_better=True,
                    description="One if a labeled original-text hit was retrieved within top_k.",
                ),
                FeedbackDefinition(
                    feedback_definition_id="ollama_eval_mrr_at_k_v1",
                    name="MRR@k",
                    higher_is_better=True,
                    description="Reciprocal of the first labeled original-text rank, or zero.",
                ),
            )
            # FeedbackResult's SQL foreign key is non-null, and the dashboard
            # uses these definitions for names and score direction.
            for feedback in feedbacks:
                session.connector.add_feedback_definition(feedback)

            for result_index, (result, scores) in enumerate(
                zip(results, relevance_scores, strict=True), start=1
            ):
                configuration = {
                    "demo": result.demo,
                    "variant": result.variant,
                    "embedding_model": result.embedding_model,
                    "chat_model": result.chat_model,
                    "judge_model": judge_model,
                    "top_k": result.top_k,
                }
                app = AppDefinition(
                    app_name=f"Ollama / {result.demo}",
                    app_version=(
                        f"variant={result.variant!r}, "
                        f"embedding={result.embedding_model!r}, "
                        f"chat={result.chat_model!r}, judge={judge_model!r}, "
                        f"top_k={result.top_k}"
                    ),
                    metadata=configuration,
                    app=configuration,
                    # The persisted app is a configuration dictionary, not an
                    # instrumented runtime; native views still require its class.
                    root_class=Class.of_class(dict, with_bases=True),
                    feedback_mode=FeedbackMode.NONE,
                    feedbacks=feedbacks,
                )
                session.add_app(app)
                for query_index, (query, relevance) in enumerate(
                    zip(result.queries, scores, strict=True), start=1
                ):
                    stored_at = datetime.now(timezone.utc)
                    rank = query.first_relevant_rank
                    hit = rank is not None and rank <= result.top_k
                    chunks = _retrieved_chunks(query)
                    events = (
                        *(result.index_trace if query_index == 1 else ()),
                        *query.trace_spans,
                    )
                    model_events = [
                        event
                        for event in events
                        if event.method
                        in {"_embeddings_for", "_generate_context_for_chunk"}
                    ]
                    usage = [
                        event.rets.get("usage")
                        for event in model_events
                        if isinstance(event.rets, dict)
                        and event.rets.get("usage") is not None
                    ]
                    prompt_tokens = sum(
                        item.get("prompt_tokens") or 0 for item in usage
                    )
                    completion_tokens = sum(
                        item.get("completion_tokens") or 0 for item in usage
                    )
                    total_tokens = sum(
                        item.get("total_tokens")
                        if item.get("total_tokens") is not None
                        else (item.get("prompt_tokens") or 0)
                        + (item.get("completion_tokens") or 0)
                        for item in usage
                    )
                    calls = [
                        RecordAppCall(
                            stack=[
                                RecordAppCallMethod(
                                    path=Lens(),
                                    method=Method(
                                        obj=Obj(
                                            cls=Class.of_class(
                                                Retriever, with_bases=True
                                            ),
                                            id=event.retriever_id,
                                        ),
                                        name=event.method,
                                    ),
                                )
                            ],
                            args=event.args,
                            rets=event.rets,
                            perf=Perf(
                                start_time=event.start_time,
                                end_time=event.end_time,
                            ),
                            pid=os.getpid(),
                            tid=get_native_id(),
                        )
                        for event in events
                    ]
                    record = Record(
                        record_id=f"record_{run_id}_{result_index}_{query_index}",
                        app_id=app.app_id,
                        main_input=query.query,
                        main_output=json.dumps(
                            {
                                "retrieved_chunks": chunks,
                                "scores": {
                                    "context_relevance": relevance,
                                    "hit_at_k": float(hit),
                                    "mrr_at_k": 1.0 / rank if hit else 0.0,
                                },
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        tags=run_id,
                        ts=stored_at,
                        meta={
                            **configuration,
                            "run_id": run_id,
                            "run_started_at": run_started_at,
                            "query_index": query_index,
                            "index_seconds": result.index_seconds,
                            "latency_seconds": query.latency_seconds,
                            "first_relevant_rank": rank,
                            "relevant_phrases": list(
                                getattr(query, "relevant_phrases", ())
                            ),
                            "context_relevance": relevance,
                            "hit_at_k": float(hit),
                            "mrr_at_k": 1.0 / rank if hit else 0.0,
                            "retrieved_chunks": chunks,
                            "trace_span_count": len(calls),
                            "trace_span_methods": [event.method for event in events],
                            "model_token_usage_available": any(
                                isinstance(event.rets, dict)
                                and event.rets.get("usage") is not None
                                for event in model_events
                            ),
                        },
                        cost=Cost(
                            n_requests=len(model_events),
                            n_successful_requests=len(model_events),
                            n_completion_requests=sum(
                                event.method == "_generate_context_for_chunk"
                                for event in model_events
                            ),
                            n_embedding_requests=sum(
                                event.method == "_embeddings_for"
                                for event in model_events
                            ),
                            n_embeddings=sum(
                                event.rets.get("embedding_count", 0)
                                for event in model_events
                                if event.method == "_embeddings_for"
                                and isinstance(event.rets, dict)
                            ),
                            n_prompt_tokens=prompt_tokens,
                            n_completion_tokens=completion_tokens,
                            n_tokens=total_tokens,
                        ),
                        perf=Perf(
                            start_time=min(
                                (event.start_time for event in events),
                                default=stored_at
                                - timedelta(seconds=query.latency_seconds),
                            ),
                            end_time=max(
                                (event.end_time for event in events),
                                default=stored_at,
                            ),
                        ),
                        calls=calls,
                    )
                    session.add_record(record)
                    values = (relevance, float(hit), 1.0 / rank if hit else 0.0)
                    for feedback, value in zip(feedbacks, values, strict=True):
                        session.add_feedback(
                            FeedbackResult(
                                record_id=record.record_id,
                                feedback_definition_id=feedback.feedback_definition_id,
                                name=feedback.name,
                                status=FeedbackResultStatus.DONE,
                                result=value,
                                calls=[
                                    FeedbackCall(
                                        args={
                                            "question": query.query,
                                            "original_snippets": list(
                                                query.original_snippets
                                            ),
                                            "first_relevant_rank": rank,
                                            "top_k": result.top_k,
                                        },
                                        ret=value,
                                        meta={"judge_model": judge_model},
                                    )
                                ],
                            )
                        )
        finally:
            TruSession.delete_singleton_by_name(session_name)
            engine.dispose()
    finally:
        if previous_tracing is None:
            os.environ.pop("TRULENS_OTEL_TRACING", None)
        else:
            os.environ["TRULENS_OTEL_TRACING"] = previous_tracing
