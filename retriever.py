import logging
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from pathlib import Path

import faiss
import numpy as np
from model_provider import create_model_client


class ChunkerKind(str, Enum):
    UNSTRUCTURED = "unstructured"
    DOCLING = "docling"


class ContextFailurePolicy(str, Enum):
    RAISE = "raise"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    chunks: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    original_text: str
    document_index: int
    source: Path | None = None
    score: float | None = None


@dataclass(frozen=True)
class _TraceEvent:
    method: str
    args: dict
    rets: object
    retriever_id: int
    start_time: datetime
    end_time: datetime


class Retriever:
    _token_pattern = re.compile(r"\w+", re.UNICODE)

    def __init__(
        self,
        documents,
        embedding_model,
        llm_model=None,
        chunker: ChunkerKind = ChunkerKind.UNSTRUCTURED,
        use_hybrid=True,
        use_reranking=True,
        use_contextual_retrieval=True,
        context_failure_policy: ContextFailurePolicy = ContextFailurePolicy.RAISE,
        openai_client=None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.documents = list(documents)
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        self.use_contextual_retrieval = use_contextual_retrieval
        self.context_failure_policy = context_failure_policy
        self._openai = openai_client
        self._reranker_model = reranker_model
        self._indexed_chunks: list[RetrievedChunk] = []
        self._index = None
        self._bm25 = None
        self._reranker = None
        self._docling_converter = None
        self._trace_callback = None

    def _trace_start(self):
        if self._trace_callback is None:
            return None
        return id(self), datetime.now(timezone.utc)

    def _trace_end(self, method, args, rets, started):
        if started is None:
            return
        retriever_id, start_time = started
        self._trace_callback(
            _TraceEvent(
                method=method,
                args=args,
                rets=rets,
                retriever_id=retriever_id,
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
            )
        )

    @staticmethod
    def _chunk_trace(chunk: RetrievedChunk) -> dict:
        return {
            "text": chunk.text,
            "original_text": chunk.original_text,
            "document_index": chunk.document_index,
            "source": str(chunk.source) if chunk.source is not None else None,
            "score": chunk.score,
        }

    @staticmethod
    def _usage_trace(usage):
        if usage is None:
            return None
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def prepare(self):
        started = self._trace_start()
        self._rebuild_from_documents(self.documents)
        self._trace_end(
            "prepare",
            {
                "document_count": len(self.documents),
                "chunker": self.chunker.value,
            },
            {
                "indexed_chunk_count": len(self._indexed_chunks),
                "embedding_model": self.embedding_model,
                "chat_model": self.llm_model,
                "contextual_retrieval": self.use_contextual_retrieval,
                "hybrid": self.use_hybrid,
                "reranking": self.use_reranking,
            },
            started,
        )
        return self

    def retrieve(self, query: str, num_neighbors: int = 2) -> list[RetrievedChunk]:
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be greater than zero")
        if not self.is_ready:
            raise RuntimeError(
                "retriever has no searchable chunks; call prepare() with supported documents"
            )

        started = self._trace_start()
        candidate_count = min(self._index.ntotal, max(num_neighbors, num_neighbors * 5))
        query_vector = self._embedding_for(query)
        vector_indices = self._indices_for_vector(query_vector, candidate_count)
        combined_indices = vector_indices

        if self.use_hybrid and self._bm25 is not None:
            tokenized_query = self._tokenize(query)
            bm25_started = self._trace_start()
            bm25_scores = self._bm25.get_scores(tokenized_query)
            bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_count]
            hybrid_ranking = [
                {"chunk_index": int(index), "score": float(bm25_scores[index])}
                for index in bm25_indices
            ]
            self._trace_end(
                "_bm25_rank",
                {"query": query, "candidate_count": candidate_count},
                hybrid_ranking,
                bm25_started,
            )
            rrf_started = self._trace_start()
            combined_indices = self._rrf_combine([vector_indices, bm25_indices])
            self._trace_end(
                "_rrf_combine",
                {
                    "vector_indices": vector_indices,
                    "bm25_ranking": hybrid_ranking,
                },
                {"chunk_indices": combined_indices},
                rrf_started,
            )

        if self.use_reranking and combined_indices:
            candidate_texts = [
                self._indexed_chunks[index].text for index in combined_indices
            ]
            pairs = [[query, text] for text in candidate_texts]
            rerank_started = self._trace_start()
            scores = self._get_reranker().predict(pairs)
            score_order = np.argsort(scores)[::-1]
            self._trace_end(
                "_rerank_candidates",
                {"query": query, "candidates": candidate_texts},
                [
                    {
                        "chunk_index": combined_indices[position],
                        "score": float(scores[position]),
                    }
                    for position in score_order
                ],
                rerank_started,
            )
            results = [
                replace(
                    self._indexed_chunks[combined_indices[position]],
                    score=float(scores[position]),
                )
                for position in score_order[:num_neighbors]
            ]
        else:
            results = [
                self._indexed_chunks[index]
                for index in combined_indices[:num_neighbors]
            ]

        self._trace_end(
            "retrieve",
            {
                "query": query,
                "num_neighbors": num_neighbors,
                "candidate_count": candidate_count,
                "embedding_model": self.embedding_model,
                "contextual_retrieval": self.use_contextual_retrieval,
                "hybrid": self.use_hybrid,
                "reranking": self.use_reranking,
            },
            [self._chunk_trace(chunk) for chunk in results],
            started,
        )
        return results

    @staticmethod
    def _rrf_combine(rankings, k=60) -> list[int]:
        scores: dict[int, float] = {}
        for ranking in rankings:
            seen = set()
            for rank, raw_index in enumerate(ranking):
                index = int(raw_index)
                if index < 0 or index in seen:
                    continue
                seen.add(index)
                scores[index] = scores.get(index, 0) + 1 / (k + rank + 1)
        return sorted(scores, key=scores.__getitem__, reverse=True)

    def _indices_for_vector(self, vector, num_neighbors: int) -> list[int]:
        started = self._trace_start()
        query = np.asarray(vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indices = self._index.search(
            query, min(num_neighbors, self._index.ntotal)
        )
        results = [int(index) for index in indices[0] if index >= 0]
        self._trace_end(
            "_indices_for_vector",
            {
                "neighbor_count": num_neighbors,
                "vector_dimensions": int(query.shape[1]),
            },
            [
                {"chunk_index": int(index), "score": float(score)}
                for score, index in zip(scores[0], indices[0], strict=True)
                if index >= 0
            ],
            started,
        )
        return results

    def add_documents(self, documents):
        new_documents = list(documents)
        start_index = len(self.documents)
        self.documents.extend(new_documents)
        self._append_documents(new_documents, start_index=start_index)
        return self

    def _rebuild_from_documents(self, documents):
        self._indexed_chunks = []
        self._index = None
        self._bm25 = None
        self._append_documents(documents, start_index=0)

    def _append_documents(self, documents, start_index: int):
        for document_index, document in enumerate(documents, start=start_index):
            parsed = self._parse_document(document)
            if not parsed.chunks:
                continue

            indexed_chunks = []
            for chunk_text in parsed.chunks:
                final_text = chunk_text
                if self.use_contextual_retrieval and self.llm_model:
                    context = self._generate_context_for_chunk(parsed.text, chunk_text)
                    if context:
                        final_text = f"{context}\n\n{chunk_text}"
                indexed_chunks.append(
                    RetrievedChunk(
                        text=final_text,
                        original_text=chunk_text,
                        document_index=document_index,
                        source=document if isinstance(document, Path) else None,
                    )
                )

            embeddings = self._embeddings_for([chunk.text for chunk in indexed_chunks])
            vectors = np.asarray(embeddings, dtype="float32")
            if vectors.ndim != 2 or vectors.shape[0] != len(indexed_chunks):
                raise ValueError("embedding API returned an unexpected vector shape")
            faiss.normalize_L2(vectors)
            if self._index is None:
                self._index = faiss.IndexFlatIP(vectors.shape[1])
            elif self._index.d != vectors.shape[1]:
                raise ValueError("embedding dimensions changed while adding documents")
            self._index.add(vectors)
            self._indexed_chunks.extend(indexed_chunks)

        if self.use_hybrid and self._indexed_chunks:
            from rank_bm25 import BM25Okapi

            tokenized_corpus = [self._tokenize(chunk.text) for chunk in self._indexed_chunks]
            self._bm25 = BM25Okapi(tokenized_corpus)

    def _parse_document(self, document: str | Path) -> ParsedDocument:
        if self.chunker == ChunkerKind.UNSTRUCTURED:
            return self._parse_with_unstructured(document)
        if self.chunker == ChunkerKind.DOCLING:
            return self._parse_with_docling(document)
        raise ValueError(f"Unsupported chunker: {self.chunker}")

    def _parse_with_unstructured(self, document: str | Path) -> ParsedDocument:
        if isinstance(document, Path):
            from unstructured.partition.auto import partition as unstructured_partition

            print("Processing document from path:", document)
            elements = unstructured_partition(filename=str(document))
        else:
            from unstructured.partition.text import partition_text as unstructured_partition_text

            elements = unstructured_partition_text(text=document)
        chunks = tuple(text for element in elements if (text := str(element).strip()))
        return ParsedDocument(text="\n\n".join(chunks), chunks=chunks)

    def _init_docling(self):
        if self._docling_converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter

            self._docling_converter = DocumentConverter(allowed_formats=[InputFormat.MD, InputFormat.PDF])

    def _parse_with_docling(self, document: str | Path) -> ParsedDocument:
        self._init_docling()
        if isinstance(document, Path):
            print("Processing document from path:", document)
            result = self._docling_converter.convert(str(document))
        else:
            from docling_core.types.io import DocumentStream

            source = DocumentStream(name="input.md", stream=BytesIO(str(document).encode("utf-8")))
            result = self._docling_converter.convert(source)
        markdown = result.document.export_to_markdown()
        chunks = tuple(chunk.strip() for chunk in markdown.split("\n\n") if chunk.strip())
        return ParsedDocument(text=markdown, chunks=chunks)

    def _generate_context_for_chunk(self, full_text: str, chunk_text: str) -> str:
        prompt = f"""
<document>
{full_text[:4000]}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""
        started = self._trace_start()
        try:
            response = self._get_model_client().chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )
            context = (response.choices[0].message.content or "").strip()
            rets = {"context": context}
            if started is not None:
                rets["usage"] = self._usage_trace(
                    getattr(response, "usage", None)
                )
            self._trace_end(
                "_generate_context_for_chunk",
                {
                    "model": self.llm_model,
                    "document": full_text[:4000],
                    "chunk": chunk_text,
                    "prompt": prompt,
                },
                rets,
                started,
            )
            return context
        except Exception:
            if self.context_failure_policy == ContextFailurePolicy.RAISE:
                raise
            logging.getLogger(__name__).exception(
                "Context generation failed; indexing the original chunk because fallback is enabled"
            )
            return ""

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self._reranker_model)
        return self._reranker

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return cls._token_pattern.findall(text.casefold())

    def _get_model_client(self):
        if self._openai is None:
            self._openai = create_model_client()
        return self._openai

    def _embedding_for(self, text):
        return self._embeddings_for([text])[0]

    def _embeddings_for(self, texts):
        started = self._trace_start()
        response = self._get_model_client().embeddings.create(
            input=texts, model=self.embedding_model
        )
        embeddings = [item.embedding for item in response.data]
        rets = {
            "embedding_count": len(embeddings),
            "vector_dimensions": len(embeddings[0]) if embeddings else 0,
        }
        if started is not None:
            rets["usage"] = self._usage_trace(getattr(response, "usage", None))
        self._trace_end(
            "_embeddings_for",
            {"model": self.embedding_model, "inputs": list(texts)},
            rets,
            started,
        )
        return embeddings

