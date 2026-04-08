from io import BytesIO
from pathlib import Path

import faiss
import numpy as np
import openai
from enum import Enum

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from unstructured.partition.auto import partition as unstructured_partition
from unstructured.partition.text import partition_text as unstructured_partition_text


class ChunkerKind(str, Enum):
    UNSTRUCTURED = "unstructured"
    DOCLING = "docling"


class Retriever:
    def __init__(self, documents, embedding_model, chunker: ChunkerKind = ChunkerKind.UNSTRUCTURED):
        self.documents = documents
        self.chunker = chunker
        self.embedding_model = embedding_model
        self._chunks = []
        self._chunk_texts = []
        self._index = None
        self._docling_converter = None

    def prepare(self):
        self._rebuild_from_documents(self.documents)

    def retrieve(self, query, num_neighbors=2):
        query_vector = self._embedding_for(query)
        return self._chunks_for_vector(query_vector, num_neighbors)

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

            embeddings = self._embeddings_for(chunk_texts)
            vectors = np.array(embeddings).astype("float32")
            self._chunks.extend(chunks)
            self._chunk_texts.extend(chunk_texts)
            if self._index is None:
                self._index = faiss.IndexFlatL2(vectors.shape[1])
            self._index.add(vectors)

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
