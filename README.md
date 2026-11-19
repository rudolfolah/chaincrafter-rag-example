# chaincrafter Retrieval Augmented Generation
Example of using [Unstructured](https://github.com/Unstructured-IO/unstructured) and [Docling](https://github.com/docling-project/docling) with [chaincrafter](https://github.com/rudolfolah/chaincrafter) and [FAISS](https://github.com/facebookresearch/faiss) to build a Retrieval Augmented Generation pipeline.

## Local model backends

### Ollama

Start `ollama serve` in another terminal if it isn't already running, then:

```shell
uv sync --locked
ollama pull qwen2.5:0.5b
ollama pull all-minilm:22m
export RAG_PROVIDER=ollama
uv run python main.py
```

Ollama doesn't require an OpenAI key. The app uses `http://127.0.0.1:11434` by default; set `OLLAMA_HOST` to use another server. The examples default to `qwen2.5:0.5b` for chat and `all-minilm:22m` for embeddings. These small models are useful for quick local checks; for larger documents or better answers, pull other models and set `RAG_CHAT_MODEL` and `RAG_EMBEDDING_MODEL`.

### LM Studio

The default provider is OpenAI-compatible. For LM Studio, load the chat model used by the example and `text-embedding-nomic-embed-text-v1.5` for embeddings. The same `RAG_CHAT_MODEL` and `RAG_EMBEDDING_MODEL` overrides work here.

```shell
uv sync
export RAG_PROVIDER=openai
export OPENAI_API_KEY='openai-api-key'
export OPENAI_BASE_URL='http://localhost:1234/v1'
uv run python main.py
```

Unstructured requires `libmagic` for detecting the file type of the documents, install it on MacOS with `brew install libmagic`

## Project Structure
- [`PromptBuilder` in `prompt_builder.py`](./prompt_builder.py) builds system and user prompts
- [`retriever.py`](./retriever.py) provides structured retrieval results with source provenance
- [`RetrievalAugmentedGenerator` in `retrieval_augmented_generator.py`](./retrieval_augmented_generator.py) returns `GeneratedAnswer` values containing messages and retrieved chunks

## Programmatic API

`run()`, `generate()`, and `answer()` return results without printing:

```python
answers = rag.run()
for answer in answers:
    print(answer.query, answer.text)
    for chunk in answer.retrieved_chunks:
        print(chunk.source, chunk.original_text)
```

## Testing

The default suite is offline and doesn't start Ollama or load models:

```shell
uv run pytest -q
uv run ruff check .
```

With Ollama running and both models already pulled, run the live smoke test:

```shell
uv run pytest -q -m ollama
```

It runs real embeddings, FAISS retrieval, and one short chat completion. It skips document parsing, reranking, and contextual enrichment so it doesn't download NLP assets or run extra inference. The first run can be slower while Ollama loads the models; subsequent runs reuse them while they're warm.

## Examples
### [`main.py`](./main.py) - Documents and Queries
Processes a set of in-memory text documents using `faiss` and then processes a set of queries. The source of the documents in the example code is Wikipedia, the URLs to the specific pages are in comments of the code.

### [`qa.py`](./qa.py) - Question and Answer
Question and Answer with PDF and MarkDown documents retrieved from `./docs/`. Copy documents into `./docs/` before running.
Use the environment for your chosen backend above.

```shell
uv run python qa.py
```

### [`wiki.py`](./wiki.py) - LLM Wiki
Maintains a persistent, compounding wiki in `./wiki/`. It supports ingesting new sources, answering queries, and performing health checks. Ensure `./wiki/` exists and contains a `SCHEMA.md` before running.
Use the environment for your chosen backend above.

```shell
uv run python wiki.py
```

Sample interactions:
- `What is systems thinking?` (Answered using the wiki and automatically filed back)
- `/ingest docs/research.pdf` (Ingests a new source into the wiki)
- `/lint` (Performs a health check for contradictions and gaps)

## [`eval_phoenix.py`](./eval_phoenix.py) - Evaluation
Runs an evaluation of the answers from a set of documents and queries, based on [`main.py`](./main.py).
Its separate `gpt-4o` judge still needs an OpenAI-compatible endpoint that serves that model, even when the RAG pipeline uses Ollama.
