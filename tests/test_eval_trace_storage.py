import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("trulens.core")

from eval_ollama import EvalResult, QueryResult
from ollama_eval_storage import save_results
from retriever import RetrievedChunk, _TraceEvent


def test_saved_records_include_retrieval_spans_and_full_chunk_evidence(tmp_path):
    started = datetime.now(timezone.utc)
    retriever_id = 12345
    index_span = _TraceEvent(
        method="_embeddings_for",
        args={"model": "tiny-embed", "inputs": ["original evidence"]},
        rets={
            "embedding_count": 1,
            "vector_dimensions": 2,
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        },
        retriever_id=retriever_id,
        start_time=started,
        end_time=started + timedelta(milliseconds=5),
    )
    context_span = _TraceEvent(
        method="_generate_context_for_chunk",
        args={
            "model": "tiny-chat",
            "document": "original evidence",
            "chunk": "original evidence",
            "prompt": "context prompt",
        },
        rets={
            "context": "Generated context",
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        },
        retriever_id=retriever_id,
        start_time=started + timedelta(milliseconds=5),
        end_time=started + timedelta(milliseconds=10),
    )
    query_span = _TraceEvent(
        method="retrieve",
        args={"query": "question", "num_neighbors": 1, "hybrid": False},
        rets=[
            {
                "text": "Generated context\n\noriginal evidence",
                "original_text": "original evidence",
                "document_index": 0,
                "source": None,
                "score": 0.9,
            }
        ],
        retriever_id=retriever_id,
        start_time=started + timedelta(milliseconds=10),
        end_time=started + timedelta(milliseconds=15),
    )
    query = QueryResult(
        query="question",
        first_relevant_rank=1,
        original_snippets=("original evidence",),
        latency_seconds=0.005,
        retrieved_chunks=(
            RetrievedChunk(
                text="Generated context\n\noriginal evidence",
                original_text="original evidence",
                document_index=0,
                score=0.9,
            ),
        ),
        relevant_phrases=("original evidence",),
        trace_spans=(query_span,),
    )
    result = EvalResult(
        demo="contextual",
        variant="enabled",
        embedding_model="tiny-embed",
        chat_model="tiny-chat",
        hit_rate=1.0,
        mrr=1.0,
        top_k=1,
        index_seconds=0.005,
        mean_query_seconds=0.005,
        queries=(query,),
        index_trace=(index_span, context_span),
    )
    path = tmp_path / "trace.sqlite"

    save_results(path, [result], [(0.75,)], judge_model="tiny-judge")

    with sqlite3.connect(path) as database:
        (record_json, cost_json) = database.execute(
            "select record_json, cost_json from trulens_records"
        ).fetchone()
    record = json.loads(record_json)
    spans = record["calls"]

    assert [span["stack"][0]["method"]["name"] for span in spans] == [
        "_embeddings_for",
        "_generate_context_for_chunk",
        "retrieve",
    ]
    assert spans[0]["args"]["inputs"] == ["original evidence"]
    assert spans[1]["args"]["prompt"] == "context prompt"
    assert spans[1]["rets"]["context"] == "Generated context"
    assert spans[2]["rets"][0]["text"] == "Generated context\n\noriginal evidence"
    assert spans[2]["rets"][0]["original_text"] == "original evidence"
    assert record["meta"]["relevant_phrases"] == ["original evidence"]
    assert record["meta"]["context_relevance"] == 0.75
    assert record["meta"]["trace_span_count"] == 3
    cost = json.loads(cost_json)
    assert cost["n_embedding_requests"] == 1
    assert cost["n_embeddings"] == 1
    assert cost["n_completion_requests"] == 1
    assert cost["n_prompt_tokens"] == 8
    assert cost["n_completion_tokens"] == 2
    assert cost["n_tokens"] == 10
