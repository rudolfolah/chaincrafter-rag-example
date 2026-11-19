from types import SimpleNamespace

import pytest

from ollama_eval_trulens import context_relevance_scores


def query_result(query, *original_snippets):
    return SimpleNamespace(query=query, original_snippets=original_snippets)


def eval_result(*queries):
    return SimpleNamespace(demo="test demo", variant="baseline", queries=queries)


class RecordingJudge:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def __call__(self, query, original_chunk):
        pair = (query, original_chunk)
        self.calls.append(pair)
        return self.scores[pair]


def test_scores_average_original_chunks_and_cache_pairs_across_results():
    judge = RecordingJudge(
        {
            ("first question", "original shared"): 0.2,
            ("first question", "original other"): 0.8,
            ("second question", "original shared"): 0.9,
        }
    )
    first = query_result(
        "first question", "original shared", "original other", "original shared"
    )
    first.enriched_snippets = ("enriched text must not be judged",)
    results = [
        eval_result(first, query_result("second question", "original shared")),
        eval_result(
            query_result("first question", "original other", "original shared"),
            query_result("second question", "original shared", "original shared"),
        ),
    ]

    scores = context_relevance_scores(results, judge)

    assert isinstance(scores, list)
    assert all(isinstance(row, tuple) for row in scores)
    assert scores[0] == pytest.approx((0.4, 0.9))
    assert scores[1] == pytest.approx((0.5, 0.9))
    assert judge.calls == [
        ("first question", "original shared"),
        ("first question", "original other"),
        ("second question", "original shared"),
    ]


def test_empty_queries_and_results_preserve_alignment_without_judging():
    judge = RecordingJudge({})

    assert context_relevance_scores([], judge) == []
    assert context_relevance_scores(
        [eval_result(), eval_result(query_result("no retrieved chunks"))], judge
    ) == [(), (0.0,)]
    assert judge.calls == []


def test_cache_is_scoped_to_one_scoring_run():
    judge = RecordingJudge({("question", "original"): 0.0})
    results = [eval_result(query_result("question", "original"))]

    assert context_relevance_scores(results, judge) == [(0.0,)]
    judge.scores[("question", "original")] = 1.0
    assert context_relevance_scores(results, judge) == [(1.0,)]
    assert judge.calls == [("question", "original"), ("question", "original")]


@pytest.mark.parametrize(
    "score", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1]
)
def test_invalid_feedback_is_rejected_with_query_and_chunk_context(score):
    judge = RecordingJudge({("invalid question", "invalid original chunk"): score})
    results = [eval_result(query_result("invalid question", "invalid original chunk"))]

    with pytest.raises(RuntimeError) as error:
        context_relevance_scores(results, judge)

    message = str(error.value)
    assert isinstance(error.value.__cause__, ValueError)
    assert "test demo" in message
    assert "baseline" in message
    assert "invalid question" in message
    assert "invalid original chunk" in message
    assert judge.calls == [("invalid question", "invalid original chunk")]


def test_judge_failure_is_not_dropped_and_retains_context_and_cause():
    failure = RuntimeError("Ollama judge unavailable")
    calls = []

    def failing_judge(query, original_chunk):
        calls.append((query, original_chunk))
        raise failure

    results = [eval_result(query_result("failed question", "failed original chunk"))]

    with pytest.raises(RuntimeError) as error:
        context_relevance_scores(results, failing_judge)

    assert "failed question" in str(error.value)
    assert "failed original chunk" in str(error.value)
    assert error.value.__cause__ is failure
    assert calls == [("failed question", "failed original chunk")]
