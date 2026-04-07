# chaincrafter Retrieval Augmentated Generation
Example of using [Unstructured](https://github.com/Unstructured-IO/unstructured) and [Docling](https://github.com/docling-project/docling) with [chaincrafter](https://github.com/rudolfolah/chaincrafter) and [faiss](https://github.com/facebookresearch/faiss) to build a Retrieval Augmentated Generation pipeline.

Tested with LM Studio running as locally hosted LLM server, for embeddings you will need the Nomic model: `text-embedding-nomic-embed-text-v1.5`.

```shell
uv sync
export OPENAI_API_KEY='openai-api-key'
export OPENAI_BASE_URL='http://localhost:1234/v1'
uv run python main.py
uv run python eval_phoenix.py
```

## Project Structure
- [`PromptBuilder` in `prompt_builder.py`](./prompt_builder.py) builds system and user prompts
- [`retriever.py`](./retriever.py) provides the `Retriever` class for chunking documents and retrieving documents
- [`RetrievalAugmentedGenerator` in `retrieval_augmented_generator.py`](./retrieval_augmented_generator.py) prepares and then generates answers to queries

## Examples
### [`main.py`](./main.py) - Documents and Queries
Processes a set of in-memory text documents using `faiss` and then processes a set of queries. The source of the documents in the example code is Wikipedia, the URLs to the specific pages are in comments of the code.

### [`qa.py`](./qa.py) - Question and Answer
Question and Answer with PDF and MarkDown documents retrieved from `./docs/`. Copy documents into `./docs/` before running.

```shell
export OPENAI_API_KEY='openai-api-key'
export OPENAI_BASE_URL='http://localhost:1234/v1'

uv run python qa.py
```

Unstructured requires `libmagic` for detecting the file type of the documents, install it on MacOS with `brew install libmagic`

## [`eval_phoenix.py`](./eval_phoenix.py) - Evaluation
Runs an evaluation of the answers from a set of documents and queries, based on [`main.py`](./main.py).
