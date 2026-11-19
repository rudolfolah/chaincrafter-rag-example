import os


def _provider() -> str:
    provider = os.getenv("RAG_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "ollama"}:
        raise ValueError(f"Unsupported RAG_PROVIDER: {provider!r}")
    return provider


def create_model_client():
    provider = _provider()
    from openai import OpenAI

    if provider == "openai":
        return OpenAI()

    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip().rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    base_url = host if host.endswith("/v1") else f"{host}/v1"
    return OpenAI(
        base_url=base_url,
        api_key="ollama",
        timeout=60.0,
        max_retries=0,
    )


def model_names(default_chat_model: str) -> tuple[str, str]:
    if _provider() == "ollama":
        chat_model = "qwen2.5:0.5b"
        embedding_model = "all-minilm:22m"
    else:
        chat_model = default_chat_model
        embedding_model = "text-embedding-nomic-embed-text-v1.5"
    return (
        os.getenv("RAG_CHAT_MODEL", chat_model),
        os.getenv("RAG_EMBEDDING_MODEL", embedding_model),
    )
