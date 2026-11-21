"""Compare retrieval demos with hand labels and local TruLens context relevance.

Run ``uv run --extra trulens python eval_ollama.py --details`` for original
retrieved snippets, per-query ranks, and TruLens scores.
Pass ``--trulens-db .trulens/ollama.sqlite`` to persist results for the native
TruLens browser dashboard; the runner prints its isolated viewer command.
Ollama must already be running with the requested models
installed; the runner never starts services or pulls Ollama models. Reranking
uses the demo's fixed sentence-transformers CrossEncoder, not an Ollama chat
model.
"""

import argparse
import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from model_provider import create_model_client
from ollama_eval_cases import DEMO_CASES, DemoCase, Judgment
from ollama_eval_trulens import context_relevance_scores, make_context_relevance
from retriever import RetrievedChunk, Retriever


DEFAULT_CHAT_MODELS = ("qwen2.5:0.5b", "smollm2:135m", "smollm2:360m")
DEFAULT_EMBEDDING_MODELS = (
    "all-minilm:22m",
    "snowflake-arctic-embed:22m",
    "nomic-embed-text",
)
DEFAULT_CONTEXTUAL_EMBEDDING_MODEL = "all-minilm:22m"
DEFAULT_JUDGE_MODEL = "gemma3:4b"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class QueryResult:
    query: str
    first_relevant_rank: int | None
    original_snippets: tuple[str, ...]
    latency_seconds: float
    retrieved_chunks: tuple[RetrievedChunk, ...] = ()
    relevant_phrases: tuple[str, ...] = ()
    trace_spans: tuple[object, ...] = ()


@dataclass(frozen=True)
class EvalResult:
    demo: str
    variant: str
    embedding_model: str
    chat_model: str | None
    hit_rate: float
    mrr: float
    top_k: int
    index_seconds: float
    mean_query_seconds: float
    queries: tuple[QueryResult, ...]
    index_trace: tuple[object, ...] = ()


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def relevant_rank(chunks: Sequence[RetrievedChunk], judgment: Judgment) -> int | None:
    """Return the first one-based original-text hit, never a generated-context hit."""
    phrases = tuple(_normalize(phrase) for phrase in judgment.relevant_phrases)
    for rank, chunk in enumerate(chunks, start=1):
        original = _normalize(chunk.original_text)
        if any(phrase in original for phrase in phrases):
            return rank
    return None


def _evaluate_variant(
    case: DemoCase,
    *,
    variant: str,
    embedding_model: str,
    chat_model: str | None,
    client,
) -> EvalResult:
    enabled = variant == "enabled"
    retriever = Retriever(
        documents=case.documents,
        embedding_model=embedding_model,
        llm_model=chat_model,
        use_contextual_retrieval=enabled and case.name == "contextual",
        use_hybrid=enabled and case.name == "hybrid",
        use_reranking=enabled and case.name == "reranking",
        openai_client=client,
        reranker_model=RERANKER_MODEL,
    )
    trace_events = []
    retriever._trace_callback = trace_events.append
    try:
        start = perf_counter()
        retriever.prepare()
        index_seconds = perf_counter() - start
        index_trace = tuple(trace_events)
        trace_events.clear()
        queries = []
        for judgment in case.judgments:
            start = perf_counter()
            chunks = retriever.retrieve(judgment.query, num_neighbors=case.top_k)
            latency = perf_counter() - start
            queries.append(
                QueryResult(
                    query=judgment.query,
                    first_relevant_rank=relevant_rank(chunks, judgment),
                    original_snippets=tuple(chunk.original_text for chunk in chunks),
                    latency_seconds=latency,
                    retrieved_chunks=tuple(chunks),
                    relevant_phrases=judgment.relevant_phrases,
                    trace_spans=tuple(trace_events),
                )
            )
            trace_events.clear()
    except Exception as exc:
        raise RuntimeError(
            f"{case.name}/{variant} (embedding={embedding_model}, "
            f"chat={chat_model or '-'}): {exc}"
        ) from exc

    count = len(queries)
    return EvalResult(
        demo=case.name,
        variant=variant,
        embedding_model=embedding_model,
        chat_model=chat_model,
        hit_rate=sum(query.first_relevant_rank is not None for query in queries)
        / count,
        mrr=sum(
            1 / query.first_relevant_rank
            for query in queries
            if query.first_relevant_rank is not None
        )
        / count,
        top_k=case.top_k,
        index_seconds=index_seconds,
        mean_query_seconds=sum(query.latency_seconds for query in queries) / count,
        queries=tuple(queries),
        index_trace=index_trace,
    )


def evaluate(
    cases: Sequence[DemoCase],
    *,
    chat_models: Sequence[str],
    embedding_models: Sequence[str],
    contextual_embedding_model: str,
    client,
) -> list[EvalResult]:
    """Evaluate paired demo variants using one caller-owned model client.

    Contextual retrieval has one vector baseline and one enabled row per chat
    model. Each hybrid/reranking embedding has its own vector baseline. No chat
    model is supplied to any baseline, hybrid, or reranking retriever.
    """
    results = []
    for case in cases:
        if case.name not in DEMO_CASES:
            raise ValueError(f"Unknown demo: {case.name!r}")
        if not case.judgments or case.top_k <= 0:
            raise ValueError(f"{case.name} requires judgments and a positive top_k")
        models = (
            (contextual_embedding_model,)
            if case.name == "contextual"
            else embedding_models
        )
        for embedding_model in models:
            results.append(
                _evaluate_variant(
                    case,
                    variant="baseline",
                    embedding_model=embedding_model,
                    chat_model=None,
                    client=client,
                )
            )
            enabled_chat_models = chat_models if case.name == "contextual" else (None,)
            for chat_model in enabled_chat_models:
                results.append(
                    _evaluate_variant(
                        case,
                        variant="enabled",
                        embedding_model=embedding_model,
                        chat_model=chat_model,
                        client=client,
                    )
                )
    return results


def _print_results(
    results: Sequence[EvalResult],
    *,
    details: bool,
    relevance_scores: Sequence[tuple[float, ...]],
    judge_model: str,
    judge_seconds: float,
) -> None:
    print("Hand relevance: ANY labeled phrase in original chunk text.")
    print(
        "Hit@k = fraction with a hit; MRR@k = mean reciprocal first-hit rank (miss = 0)."
    )
    print(f"TruLens Context Relevance judge: {judge_model} (Ollama only).")
    print(
        "TruLens: built-in rubric, normalized [0,1]; mean original-chunk score "
        "per query, then mean queries per variant; empty retrieval = 0."
    )
    print(
        "Independent of hand labels; generation/style requests are ignored. "
        "Duplicate query/chunk pairs are scored once."
    )
    print(
        "Times: index = prepare(); query = mean retrieve(), excluding judging/printing."
    )
    print(
        "Cold starts/model loading may affect timings; no warm-up or repeated trials."
    )
    print(f"TruLens judge_s={judge_seconds:.3f} (after all retrieval timings).")
    print(f"Enabled reranking uses fixed CrossEncoder: {RERANKER_MODEL}")
    print(
        "demo\tvariant\tembedding_model\tchat_model\tk\tHit@k\tMRR@k\t"
        "context_relevance\tindex_s\tmean_query_s"
    )
    for result, query_scores in zip(results, relevance_scores, strict=True):
        relevance = sum(query_scores) / len(query_scores) if query_scores else 0.0
        print(
            f"{result.demo}\t{result.variant}\t{result.embedding_model}\t"
            f"{result.chat_model or '-'}\t{result.top_k}\t{result.hit_rate:.3f}\t"
            f"{result.mrr:.3f}\t{relevance:.3f}\t{result.index_seconds:.3f}\t"
            f"{result.mean_query_seconds:.3f}"
        )
        if details:
            for query, score in zip(result.queries, query_scores, strict=True):
                rank = query.first_relevant_rank
                print(f"  Query: {query.query}")
                print(
                    f"  First relevant rank: {rank if rank is not None else 'miss'}; "
                    f"retrieve_s={query.latency_seconds:.3f}; "
                    f"context_relevance={score:.3f}"
                )
                for index, snippet in enumerate(query.original_snippets, start=1):
                    print(f"  [{index}] Original chunk: {snippet}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        nargs="+",
        choices=tuple(DEMO_CASES),
        default=list(DEMO_CASES),
        help="Demos to compare (default: all three).",
    )
    parser.add_argument(
        "--chat-models",
        nargs="+",
        default=list(DEFAULT_CHAT_MODELS),
        help="Ollama chat tags used only for enabled contextual retrieval.",
    )
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=list(DEFAULT_EMBEDDING_MODELS),
        help="Ollama embedding tags compared for hybrid and reranking.",
    )
    parser.add_argument(
        "--contextual-embedding-model",
        default=DEFAULT_CONTEXTUAL_EMBEDDING_MODEL,
        help="Fixed Ollama embedding tag for all contextual rows.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Ollama tag for the independent TruLens judge (default: gemma3:4b).",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show per-query ranks, TruLens scores, and retrieved original chunks.",
    )
    parser.add_argument(
        "--trulens-db",
        type=Path,
        help="Persist results to this TruLens SQLite database (opt-in).",
    )
    args = parser.parse_args(argv)
    previous = {
        name: os.environ.get(name) for name in ("RAG_PROVIDER", "KMP_DUPLICATE_LIB_OK")
    }
    try:
        # Use Ollama regardless of the app's configured default provider.
        os.environ["RAG_PROVIDER"] = "ollama"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        with create_model_client() as client:
            print(f"Ollama endpoint: {client.base_url}", flush=True)
            score_chunk = make_context_relevance(client, args.judge_model)
            results = evaluate(
                [DEMO_CASES[name] for name in args.demo],
                chat_models=args.chat_models,
                embedding_models=args.embedding_models,
                contextual_embedding_model=args.contextual_embedding_model,
                client=client,
            )
            start = perf_counter()
            relevance_scores = context_relevance_scores(results, score_chunk)
            judge_seconds = perf_counter() - start
        _print_results(
            results,
            details=args.details,
            relevance_scores=relevance_scores,
            judge_model=args.judge_model,
            judge_seconds=judge_seconds,
        )
        if args.trulens_db is not None:
            try:
                from ollama_eval_storage import save_results

                save_results(
                    args.trulens_db,
                    results,
                    relevance_scores,
                    judge_model=args.judge_model,
                )
            except Exception as exc:
                parser.exit(
                    1,
                    f"TruLens persistence failed for {args.trulens_db}: {exc}\n",
                )
            print(
                "TruLens dashboard: "
                + shlex.join(
                    [
                        "uv",
                        "run",
                        "--script",
                        "eval_trulens.py",
                        "--trulens-db",
                        str(args.trulens_db),
                    ]
                )
            )
    except Exception as exc:
        parser.exit(
            1,
            f"Evaluation failed: {exc}\n"
            "Ensure Ollama is running at OLLAMA_HOST (default http://127.0.0.1:11434) "
            "and the requested model tags are already installed. "
            "Reranking also requires the fixed CrossEncoder and its dependencies. "
            "No Ollama models are pulled automatically.\n",
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
