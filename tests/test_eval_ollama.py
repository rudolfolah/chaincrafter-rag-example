from collections import Counter

import pytest

import eval_ollama
from ollama_eval_cases import DemoCase, Judgment
from retriever import RetrievedChunk


def chunk(original, *, enriched=None):
    return RetrievedChunk(
        text=original if enriched is None else enriched,
        original_text=original,
        document_index=0,
    )


def test_relevant_rank_ignores_enrichment_and_normalizes_original_text():
    judgment = Judgment(query="question", relevant_phrases=("  TARGET\n phrase  ",))
    chunks = [
        chunk("irrelevant source", enriched="target phrase"),
        chunk("Before TaRgEt\t \nPHRASE after", enriched="unrelated context"),
        chunk("target phrase"),
    ]

    assert eval_ollama.relevant_rank(chunks, judgment) == 2
    assert eval_ollama.relevant_rank(chunks[:1], judgment) is None
    assert eval_ollama.relevant_rank([], judgment) is None


@pytest.mark.parametrize("phrase", ["first alternative", "second alternative"])
def test_relevant_rank_accepts_any_phrase_alternative(phrase):
    judgment = Judgment(
        query="question",
        relevant_phrases=("first alternative", "second alternative"),
    )

    assert eval_ollama.relevant_rank([chunk(f"prefix {phrase} suffix")], judgment) == 1
    assert eval_ollama.relevant_rank([chunk("neither alternative")], judgment) is None


def test_evaluate_compares_isolated_features_and_aggregates_missed_queries(monkeypatch):
    instances = []
    client = object()

    class FakeRetriever:
        def __init__(
            self,
            documents,
            embedding_model,
            llm_model=None,
            *,
            use_hybrid,
            use_reranking,
            use_contextual_retrieval,
            openai_client,
            **kwargs,
        ):
            assert openai_client is client
            self.documents = tuple(documents)
            self.embedding_model = embedding_model
            self.llm_model = llm_model
            self.flags = (use_contextual_retrieval, use_hybrid, use_reranking)
            assert sum(self.flags) <= 1
            self.enabled = any(self.flags)
            self.queries = []
            self.prepared = False
            instances.append(self)

        def prepare(self):
            assert not self.prepared
            self.prepared = True
            return self

        def retrieve(self, query, num_neighbors=2):
            assert self.prepared
            assert num_neighbors == 2
            self.queries.append(query)
            irrelevant = chunk("unrelated source", enriched="answer")
            relevant = chunk("the ANSWER is here")
            if query == "first":
                return (
                    [relevant, irrelevant] if self.enabled else [irrelevant, relevant]
                )
            if query == "second" and self.enabled:
                return [irrelevant, relevant]
            return [irrelevant, irrelevant]

    monkeypatch.setattr(eval_ollama, "Retriever", FakeRetriever)
    judgments = tuple(
        Judgment(query=query, relevant_phrases=("answer",))
        for query in ("first", "second", "missed")
    )
    cases = [
        DemoCase(name=name, documents=(f"{name} source",), judgments=judgments, top_k=2)
        for name in ("contextual", "hybrid", "reranking")
    ]

    results = eval_ollama.evaluate(
        cases,
        chat_models=("chat-a", "chat-b"),
        embedding_models=("embed-a", "embed-b"),
        contextual_embedding_model="fixed-embed",
        client=client,
    )

    expected_rows = {
        ("contextual", "baseline", "fixed-embed", None),
        ("contextual", "enabled", "fixed-embed", "chat-a"),
        ("contextual", "enabled", "fixed-embed", "chat-b"),
    }
    expected_rows.update(
        (demo, variant, embedding, None)
        for demo in ("hybrid", "reranking")
        for variant in ("baseline", "enabled")
        for embedding in ("embed-a", "embed-b")
    )
    observed_rows = [
        (result.demo, result.variant, result.embedding_model, result.chat_model)
        for result in results
    ]
    assert Counter(observed_rows) == Counter(expected_rows)
    for result in results:
        if result.variant == "baseline":
            assert result.hit_rate == pytest.approx(1 / 3)
            assert result.mrr == pytest.approx(1 / 6)
        else:
            assert result.hit_rate == pytest.approx(2 / 3)
            assert result.mrr == pytest.approx(1 / 2)

    expected_configs = Counter(
        (
            (f"{demo} source",),
            embedding,
            chat,
            tuple(
                variant == "enabled" and demo == feature
                for feature in ("contextual", "hybrid", "reranking")
            ),
        )
        for demo, variant, embedding, chat in expected_rows
    )
    assert (
        Counter(
            (
                instance.documents,
                instance.embedding_model,
                instance.llm_model,
                instance.flags,
            )
            for instance in instances
        )
        == expected_configs
    )
    assert all(
        instance.queries == ["first", "second", "missed"] for instance in instances
    )
