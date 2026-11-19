import os
import re

import pytest

from model_provider import create_model_client, model_names


def _unset_openai_env(monkeypatch):
    for name in os.environ:
        if name.startswith("OPENAI_"):
            monkeypatch.delenv(name, raising=False)


@pytest.mark.ollama
def test_ollama_retrieval_augmented_answer(monkeypatch):
    from retrieval_augmented_generator import RetrievalAugmentedGenerator
    from retriever import ParsedDocument, Retriever

    monkeypatch.setenv("RAG_PROVIDER", "ollama")
    _unset_openai_env(monkeypatch)
    chat_model, embedding_model = model_names("unused-openai-default")
    source = "The launch code is ORCHID."
    query = "What is the launch code? Reply with only the code."
    # The corpus is already one parsed chunk; avoid unrelated NLTK downloads.
    monkeypatch.setattr(
        Retriever,
        "_parse_document",
        lambda self, document: ParsedDocument(text=document, chunks=(document,)),
    )
    generator = RetrievalAugmentedGenerator(
        documents=[source],
        queries=[query],
        embedding_model=embedding_model,
        llm_model=chat_model,
        num_neighbors=1,
        temperature=0,
        max_tokens=12,
        system_prompt="Answer using only the supplied context. Reply with only the requested code.",
        use_hybrid=False,
        use_reranking=False,
        use_contextual_retrieval=False,
    )

    answers = generator.run()

    assert len(answers) == 1
    answer = answers[0]
    assert len(answer.retrieved_chunks) == 1
    assert source in answer.retrieved_chunks[0].original_text
    assert re.search(r"\borchid\b", answer.text, re.IGNORECASE), answer.text


def test_ollama_provider_ignores_openai_configuration(monkeypatch):
    monkeypatch.setenv("RAG_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:12345/")
    monkeypatch.setenv("OPENAI_API_KEY", "not-an-ollama-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.invalid/v1")
    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)

    assert model_names("existing-chat-default") == ("qwen2.5:0.5b", "all-minilm:22m")
    with create_model_client() as client:
        assert str(client.base_url) == "http://127.0.0.1:12345/v1/"
        assert client.api_key
        assert client.api_key != "not-an-ollama-key"
    assert os.environ["OPENAI_API_KEY"] == "not-an-ollama-key"
    assert os.environ["OPENAI_BASE_URL"] == "https://openai.invalid/v1"


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_provider_model_overrides(monkeypatch, provider):
    monkeypatch.setenv("RAG_PROVIDER", provider)
    monkeypatch.setenv("RAG_CHAT_MODEL", "custom-chat")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "custom-embedding")

    assert model_names("existing-chat-default") == ("custom-chat", "custom-embedding")


def test_default_provider_retains_existing_models(monkeypatch):
    monkeypatch.delenv("RAG_PROVIDER", raising=False)
    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)

    assert model_names("existing-chat-default") == (
        "existing-chat-default",
        "text-embedding-nomic-embed-text-v1.5",
    )
