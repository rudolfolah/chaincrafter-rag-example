from io import BytesIO
from pathlib import Path

import faiss
import numpy as np
import openai
from enum import Enum
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from unstructured.partition.auto import partition as unstructured_partition
from unstructured.partition.text import partition_text as unstructured_partition_text


class ChunkerKind(str, Enum):
    UNSTRUCTURED = "unstructured"
    DOCLING = "docling"


class Retriever:
    def __init__(
        self,
        documents,
        embedding_model,
        llm_model=None,
        chunker: ChunkerKind = ChunkerKind.UNSTRUCTURED,
        use_hybrid=True,
        use_reranking=True,
        use_contextual_retrieval=True,
    ):
        self.documents = documents
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        self.use_contextual_retrieval = use_contextual_retrieval
        self._chunks = []
        self._chunk_texts = []
        self._index = None
        self._bm25 = None
        self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._docling_converter = None

    def prepare(self):
        self._rebuild_from_documents(self.documents)

    def retrieve(self, query, num_neighbors=2):
        # 1. Vector search
        query_vector = self._embedding_for(query)
        vector_results_indices = self._indices_for_vector(query_vector, num_neighbors * 5)
        
        combined_indices = vector_results_indices
        
        # 2. Hybrid search (BM25 + RRF)
        if self.use_hybrid and self._bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokenized_query)
            bm25_indices = np.argsort(bm25_scores)[::-1][:num_neighbors * 5]
            combined_indices = self._rrf_combine([vector_results_indices, bm25_indices])
        
        # 3. Re-ranking
        if self.use_reranking and len(combined_indices) > 0:
            candidate_texts = [self._chunk_texts[i] for i in combined_indices]
            pairs = [[query, text] for text in candidate_texts]
            scores = self._reranker.predict(pairs)
            ranked_indices = [combined_indices[i] for i in np.argsort(scores)[::-1]]
            return [self._chunk_texts[i] for i in ranked_indices[:num_neighbors]]
        
        return [self._chunk_texts[i] for i in combined_indices[:num_neighbors]]

    def _rrf_combine(self, rankings, k=60):
        scores = {}
        for ranking in rankings:
            for rank, index in enumerate(ranking):
                scores[index] = scores.get(index, 0) + 1 / (k + rank + 1)
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    def _indices_for_vector(self, vector, num_neighbors):
        query = np.array(vector).astype("float32").reshape(1, -1)
        _, indices = self._index.search(query, num_neighbors)
        return indices[0]

    def add_documents(self, documents):
        self.documents.extend(documents)
        self._append_documents(documents)
        return self

    def _rebuild_from_documents(self, documents):
        self._chunks = []
        self._chunk_texts = []
        self._index = None
        self._append_documents(documents)

    def _append_documents(self, documents):
        for document in documents:
            chunks = self._chunks_from_document(document)
            chunk_texts = [chunk.text if hasattr(chunk, "text") else chunk for chunk in chunks]
            
            if not chunk_texts:
                continue

            # Contextual Retrieval
            if self.use_contextual_retrieval and self.llm_model:
                full_text = self._get_full_text_for_document(document)
                final_texts = []
                for chunk_text in chunk_texts:
                    context = self._generate_context_for_chunk(full_text, chunk_text)
                    final_texts.append(f"{context}\n\n{chunk_text}")
            else:
                final_texts = chunk_texts

            embeddings = self._embeddings_for(final_texts)
            vectors = np.array(embeddings).astype("float32")
            self._chunks.extend(chunks)
            self._chunk_texts.extend(final_texts)
            if self._index is None:
                self._index = faiss.IndexFlatL2(vectors.shape[1])
            self._index.add(vectors)
        
        # Update BM25 index after all documents are added
        if self._chunk_texts:
            tokenized_corpus = [doc.lower().split() for doc in self._chunk_texts]
            self._bm25 = BM25Okapi(tokenized_corpus)

    def _get_full_text_for_document(self, document):
        if isinstance(document, Path):
            return document.read_text(encoding="utf-8", errors="ignore")
        return str(document)

    def _generate_context_for_chunk(self, full_text, chunk_text):
        if not self.llm_model:
            return ""
        
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
        try:
            response = openai.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating context: {e}")
            return ""

    def _chunks_from_documents(self):
        chunks = []
        for doc in self.documents:
            chunks.extend(self._chunks_from_document(doc))
        return chunks

    def _chunks_from_document(self, document):
        if self.chunker == ChunkerKind.UNSTRUCTURED:
            return self._unstructured_chunks(document)

        if self.chunker == ChunkerKind.DOCLING:
            return self._docling_chunks(document)

        raise ValueError(f"Unsupported chunker: {self.chunker}")

    def _unstructured_chunks(self, document: str | Path) -> list[str]:
        if isinstance(document, Path):
            print("Processing document from path:", document)
            elements = unstructured_partition(document)
        else:
            elements = unstructured_partition_text(text=document)
        return [str(element) for element in elements]

    def _init_docling(self):
        if self._docling_converter is None:
            self._docling_converter = DocumentConverter(allowed_formats=[InputFormat.MD, InputFormat.PDF])

    def _docling_chunks(self, document: str | Path):
        self._init_docling()
        if isinstance(document, Path):
            print("Processing document from path:", document)
            result = self._docling_converter.convert(str(document))
        else:
            stream = BytesIO(str(document).encode("utf-8"))
            source = DocumentStream(name="input.md", stream=stream)
            result = self._docling_converter.convert(source)
        markdown = result.document.export_to_markdown()
        return [chunk.strip() for chunk in markdown.split("\n\n") if chunk.strip()]

    def _texts_from_chunks(self):
        return [chunk.text if hasattr(chunk, "text") else chunk for chunk in self._chunks]

    def _index_from_texts(self):
        if not self._chunk_texts:
            return None
        embeddings = self._embeddings_for(self._chunk_texts)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings).astype("float32"))
        return index

    def _embedding_for(self, text):
        return self._embeddings_for([text])[0]

    def _embeddings_for(self, texts):
        response = openai.embeddings.create(input=texts, model=self.embedding_model)
        return [item.embedding for item in response.data]

    def _chunks_for_vector(self, vector, num_neighbors):
        query = np.array(vector).astype("float32").reshape(1, -1)
        _, indices = self._index.search(query, num_neighbors)
        return [self._chunk_texts[i] for i in indices[0]]
