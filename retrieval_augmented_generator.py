from pathlib import Path

from chaincrafter import Chain
from chaincrafter.models import OpenAiChat

from prompt_builder import PromptBuilder
from retriever import Retriever, ChunkerKind


class RetrievalAugmentedGenerator:
    def __init__(
        self,
        documents: list[str] | list[Path],
        queries: list[str],
        embedding_model: str,
        llm_model: str,
        num_neighbors: int = 3,
        temperature: float = 0.55,
        system_prompt: str | None = None,
        chunker: ChunkerKind = ChunkerKind.UNSTRUCTURED,
        use_hybrid=True,
        use_reranking=True,
        use_contextual_retrieval=True,
    ):
        if len(documents) <= 0:
            raise ValueError("documents must be non-empty")
        self.queries = queries
        self.retriever = Retriever(
            documents,
            embedding_model,
            llm_model=llm_model,
            chunker=chunker,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
            use_contextual_retrieval=use_contextual_retrieval,
        )
        self._num_neighbors = num_neighbors
        self._retrieved_documents = {}
        system_prompt = system_prompt or "You are a helpful assistant, responding to questions about the world. " \
            "Do not pose any follow-up questions or make any suggestions. " \
            "Do not have a preamble in any response."
        self.prompts = PromptBuilder(system_prompt)
        self.chat_model = OpenAiChat(temperature=temperature, model_name=llm_model)

    def prepare(self):
        self.retriever.prepare()
        return self

    def generate(self):
        for query in self.queries:
            print(f"--- {query}")
            for message in self._answer(query):
                print(f"{message['role']}: {message['content']}")
            print("\n\n")

    def run(self):
        self.prepare()
        self.generate()

    def get_retrieved_documents(self, query):
        chunks = self.retriever.retrieve(query, num_neighbors=self._num_neighbors)
        self._retrieved_documents[query] = chunks
        return chunks

    def _answer(self, query):
        chunks = self.get_retrieved_documents(query)
        context = "\n\n".join(chunks)
        chain = Chain(
            self.prompts.system_prompt(),
            (self.prompts.user_prompt(), "answer")
        )
        return chain.run(self.chat_model, {"query": query, "context": context})
