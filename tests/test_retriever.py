import builtins
import sys
from types import SimpleNamespace

import pytest

import retriever as retriever_module
from retriever import ContextFailurePolicy, ParsedDocument, Retriever


class FakeEmbeddings:
    def create(self, input, model):
        vectors = [
            [1.0, 0.0] if "alpha" in text.casefold() else [0.0, 1.0]
            for text in input
        ]
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=vector) for vector in vectors]
        )


class FakeCompletions:
    def __init__(self, content="generated context", error=None):
        self.content = content
        self.error = error

    def create(self, **kwargs):
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class StubRetriever(Retriever):
    def _parse_document(self, document):
        text = (
            document.read_text(encoding="utf-8")
            if hasattr(document, "read_text")
            else str(document)
        )
        return ParsedDocument(text=text, chunks=(text,))


def fake_client(*, context="generated context", error=None):
    return SimpleNamespace(
        embeddings=FakeEmbeddings(),
        chat=SimpleNamespace(
            completions=FakeCompletions(content=context, error=error)
        ),
    )


def make_retriever(documents, **overrides):
    options = {
        "use_hybrid": True,
        "use_reranking": False,
        "use_contextual_retrieval": False,
        "openai_client": fake_client(),
    }
    options.update(overrides)
    return StubRetriever(documents, "fake-embedding", **options)


def test_retrieve_clamps_candidates_without_faiss_padding():
    retriever = make_retriever(["alpha", "beta"]).prepare()

    results = retriever.retrieve("alpha", num_neighbors=5)

    assert [result.text for result in results] == ["alpha", "beta"]
    assert len({result.document_index for result in results}) == 2


def test_retrieve_rejects_invalid_state_and_neighbor_count():
    retriever = make_retriever([])

    with pytest.raises(RuntimeError, match="no searchable chunks"):
        retriever.retrieve("query")

    retriever = make_retriever(["alpha"]).prepare()
    with pytest.raises(ValueError, match="greater than zero"):
        retriever.retrieve("query", num_neighbors=0)


def test_reranker_is_lazy_when_disabled(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            pytest.fail("disabled reranking must not import Sentence Transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    retriever = make_retriever(["alpha"]).prepare()
    retriever.retrieve("alpha")
    assert retriever._reranker is None


def test_contextual_chunks_keep_original_text_and_source(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("alpha", encoding="utf-8")
    retriever = make_retriever(
        [source],
        llm_model="fake-chat",
        use_contextual_retrieval=True,
        openai_client=fake_client(context="document context"),
    ).prepare()

    result = retriever.retrieve("alpha", num_neighbors=1)[0]

    assert result.text == "document context\n\nalpha"
    assert result.original_text == "alpha"
    assert result.source == source


def test_unstructured_parser_uses_extracted_pdf_text(monkeypatch, tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF raw bytes")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.auto",
        SimpleNamespace(
            partition=lambda **kwargs: calls.append(kwargs) or ["Parsed first", "Parsed second"]
        ),
    )
    retriever = make_retriever([])

    parsed = retriever._parse_with_unstructured(source)

    assert calls == [{"filename": str(source)}]
    assert parsed.text == "Parsed first\n\nParsed second"
    assert parsed.chunks == ("Parsed first", "Parsed second")


def test_context_failure_policy_is_explicit():
    error = ConnectionError("offline")
    raising = make_retriever(
        ["alpha"],
        llm_model="fake-chat",
        use_contextual_retrieval=True,
        openai_client=fake_client(error=error),
    )
    with pytest.raises(ConnectionError, match="offline"):
        raising.prepare()

    fallback = make_retriever(
        ["alpha"],
        llm_model="fake-chat",
        use_contextual_retrieval=True,
        context_failure_policy=ContextFailurePolicy.FALLBACK,
        openai_client=fake_client(error=error),
    ).prepare()
    assert fallback.retrieve("alpha")[0].text == "alpha"


def test_tokenizer_handles_case_and_punctuation_consistently():
    assert Retriever._tokenize("Hello, WORLD!") == ["hello", "world"]


def test_default_client_is_created_once_on_first_request(monkeypatch):
    client = fake_client()
    created = []
    monkeypatch.setattr(
        retriever_module, "create_model_client", lambda: created.append(client) or client
    )
    retriever = make_retriever(
        ["alpha"],
        openai_client=None,
        llm_model="fake-chat",
        use_contextual_retrieval=True,
    )
    assert created == []

    retriever.prepare()
    retriever.retrieve("alpha")

    assert created == [client]
    assert retriever._openai is client


def test_injected_client_does_not_create_default(monkeypatch):
    def unexpected_client():
        pytest.fail("an injected client must remain authoritative")

    monkeypatch.setattr(retriever_module, "create_model_client", unexpected_client)
    retriever = make_retriever(
        ["alpha"], llm_model="fake-chat", use_contextual_retrieval=True
    ).prepare()
    assert retriever.retrieve("alpha")[0].original_text == "alpha"


def test_vector_only_retrieval_does_not_import_optional_stacks(monkeypatch):
    original_import = builtins.__import__
    optional_roots = {
        "docling", "docling_core", "unstructured", "sentence_transformers", "rank_bm25"
    }

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in optional_roots:
            pytest.fail(f"vector-only retrieval imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    retriever = make_retriever(["alpha", "beta"], use_hybrid=False).prepare()

    assert retriever._bm25 is None
    assert retriever.retrieve("alpha")[0].text == "alpha"


def test_unstructured_text_does_not_import_auto_partition(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.text",
        SimpleNamespace(partition_text=lambda **kwargs: [kwargs["text"]]),
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "unstructured.partition.auto":
            pytest.fail("plain text must not import the file parser stack")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    parsed = make_retriever([])._parse_with_unstructured("alpha")

    assert parsed == ParsedDocument(text="alpha", chunks=("alpha",))
