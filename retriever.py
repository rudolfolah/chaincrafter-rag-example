import faiss
import numpy as np
import openai
from chonkie import TokenChunker
from tokenizers import Tokenizer


class Retriever:
    def __init__(self, documents, embedding_model):
        self._documents = documents
        self._chunker = TokenChunker(Tokenizer.from_pretrained("gpt2"))
        self._embedding_model = embedding_model
        self._chunks = []
        self._chunk_texts = []
        self._index = None

    def prepare(self):
        self._chunks = self._chunks_from_documents()
        self._chunk_texts = self._texts_from_chunks()
        self._index = self._index_from_texts()
        return self

    def retrieve(self, query, num_neighbors=2):
        query_vector = self._embedding_for(query)
        return self._chunks_for_vector(query_vector, num_neighbors)

    def _chunks_from_documents(self):
        return [chunk for doc in self._documents for chunk in self._chunker(doc)]

    def _texts_from_chunks(self):
        return [chunk.text for chunk in self._chunks]

    def _index_from_texts(self):
        embeddings = self._embeddings_for(self._chunk_texts)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings).astype("float32"))
        return index

    def _embeddings_for(self, texts):
        response = openai.embeddings.create(input=texts, model=self._embedding_model)
        return [item.embedding for item in response.data]

    def _embedding_for(self, text):
        return self._embeddings_for([text])[0]

    def _chunks_for_vector(self, vector, num_neighbors):
        query = np.array(vector).astype("float32").reshape(1, -1)
        _, indices = self._index.search(query, num_neighbors)
        return [self._chunk_texts[i] for i in indices[0]]
