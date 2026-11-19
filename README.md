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

## [`eval_ollama.py`](./eval_ollama.py) - Local demo comparison

With Ollama running, pull the default models and run:

```shell
for model in qwen2.5:0.5b smollm2:135m smollm2:360m all-minilm:22m \
             snowflake-arctic-embed:22m nomic-embed-text gemma3:4b; do
  ollama pull "$model" || break
done
uv run --locked --extra trulens python eval_ollama.py
```

| Demo | What changes | Baseline |
| --- | --- | --- |
| Contextual retrieval | Three Ollama chat models add context before indexing; embeddings stay fixed | Vector search without added context |
| Hybrid search | Three Ollama embedding models, each with BM25 and reciprocal rank fusion | Vector search with the same embedding model |
| Reranking | Three Ollama embedding models, each with the same CrossEncoder | Vector search with the same embedding model |

Each demo uses its existing corpus and three hand-labeled queries. The report shows Hit@k (fraction of queries with a relevant result), MRR@k (mean reciprocal rank of the first relevant result), and TruLens Context Relevance. The fixed local `gemma3:4b` judge grades original chunks on a 0–1 scale, separately from the hand labels and after all index and query timings. This small dataset measures retrieval, not generated answer quality.

Use `--details` for per-query ranks, scores, and retrieved text. Use `--demo contextual hybrid` to limit the run; `--chat-models`, `--embedding-models`, `--contextual-embedding-model`, and `--judge-model` select installed tags. A custom judge must support structured JSON output. The eval uses `OLLAMA_HOST` and needs no OpenAI key. The first run may download Unstructured tokenizer data and the fixed Hugging Face reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`); these are reused from cache.

### View the results in a browser

```shell
uv run --locked --extra trulens python eval_ollama.py --trulens-db .trulens/ollama.sqlite
uv run --script eval_trulens.py
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). Select a demo in the sidebar: **Leaderboard** compares model variants, and **Records** shows individual queries, retrieved chunks, and scores. Runs append to the database. To view saved runs again, start only the dashboard; it doesn't call Ollama.
New records include retrieval spans in TruLens **Span JSON**: contextual prompts and outputs, embedding inputs and dimensions, vector/BM25 rankings, reranking scores, and returned chunks with original and enriched text. Per-record metadata includes hand-label ranks and metrics. Token counts are populated when Ollama returns usage data; local generation cost remains $0. Older records have no spans to backfill, so run the eval again to create traced records.

For a faster first run, add `--demo hybrid` to the first command. To use a different database, pass the same `--trulens-db PATH` to both commands. The dashboard runs in its own Python 3.13 environment because its `psutil` requirement conflicts with Unstructured's.
The first recorded query for each model variant includes index preparation; subsequent records include only their own query spans, avoiding duplicate index counts.

## [`eval_phoenix.py`](./eval_phoenix.py) - Evaluation
Runs an evaluation of the answers from a set of documents and queries, based on [`main.py`](./main.py).
Its separate `gpt-4o` judge still needs an OpenAI-compatible endpoint that serves that model, even when the RAG pipeline uses Ollama.
