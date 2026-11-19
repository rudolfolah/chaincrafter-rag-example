from dataclasses import dataclass
from pathlib import Path

from chaincrafter import Chain
from chaincrafter.models import OpenAiChat

from prompt_builder import PromptBuilder
from retriever import ChunkerKind, RetrievedChunk, Retriever


@dataclass(frozen=True)
class GeneratedAnswer:
    query: str
    messages: tuple[dict, ...]
    retrieved_chunks: tuple[RetrievedChunk, ...]

    @property
    def text(self) -> str:
        return "".join(
            message["content"]
            for message in self.messages
            if message.get("role") == "assistant"
        )


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
        self._retrieved_documents: dict[str, list[RetrievedChunk]] = {}
        system_prompt = system_prompt or (
            "You are a helpful assistant, responding to questions about the world. "
            "Do not pose any follow-up questions or make any suggestions. "
            "Do not have a preamble in any response."
        )
        self.prompts = PromptBuilder(system_prompt)
        self.chat_model = OpenAiChat(temperature=temperature, model_name=llm_model)

    def prepare(self):
        self.retriever.prepare()
        return self

    def generate(self) -> list[GeneratedAnswer]:
        return [self.answer(query) for query in self.queries]

    def run(self) -> list[GeneratedAnswer]:
        self.prepare()
        return self.generate()

    def get_retrieved_documents(self, query: str) -> list[RetrievedChunk]:
        chunks = self.retriever.retrieve(query, num_neighbors=self._num_neighbors)
        self._retrieved_documents[query] = chunks
        return chunks

    def answer(self, query: str) -> GeneratedAnswer:
        chunks = self.get_retrieved_documents(query)
        context = "\n\n".join(chunk.text for chunk in chunks)
        chain = Chain(
            self.prompts.system_prompt(),
            (self.prompts.user_prompt(), "answer"),
        )
        messages = tuple(chain.run(self.chat_model, {"query": query, "context": context}))
        return GeneratedAnswer(
            query=query,
            messages=messages,
            retrieved_chunks=tuple(chunks),
        )
