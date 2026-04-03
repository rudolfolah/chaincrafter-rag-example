# chaincrafter Retrieval Augmentated Generation
Example of using [chonkie](https://github.com/bhavnicksm/chonkie) with [chaincrafter](https://github.com/rudolfolah/chaincrafter) and [faiss](https://github.com/facebookresearch/faiss) to build a Retrieval Augmentated Generation pipeline.

Tested with LM Studio running as locally hosted LLM server, for embeddings you will need the Nomic model: `text-embedding-nomic-embed-text-v1.5`.

```shell
uv sync
export OPENAI_API_KEY='openai-api-key'
export OPENAI_BASE_URL='http://localhost:1234/v1'
uv run python main.py
```

The source of the documents in the example code is Wikipedia, the URLs to the specific pages are in comments of the code.
