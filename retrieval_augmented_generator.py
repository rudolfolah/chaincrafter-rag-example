from dataclasses import dataclass
from pathlib import Path

from chaincrafter import Chain
from chaincrafter.models import ChatModel

from model_provider import create_model_client
from prompt_builder import PromptBuilder
from retriever import ChunkerKind, RetrievedChunk, Retriever


class _ClientChatModel(ChatModel):
    def __init__(
        self,
        client,
        model_name: str,
        temperature: float,
        max_tokens: int | None,
    ):
        super().__init__(temperature, model_name)
        self._client = client
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, messages):
        options = {}
        if self._max_tokens is not None:
            options["max_tokens"] = self._max_tokens
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            **options,
        )
        return response.choices[0].message.content


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
        *,
        max_tokens: int | None = None,
    ):
        self.queries = queries
        client = create_model_client()
        self.retriever = Retriever(
            documents,
            embedding_model,
            llm_model=llm_model,
            chunker=chunker,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
            use_contextual_retrieval=use_contextual_retrieval,
            openai_client=client,
        )
        self._num_neighbors = num_neighbors
        self._retrieved_documents: dict[str, list[RetrievedChunk]] = {}
        system_prompt = system_prompt or (
            "You are a helpful assistant, responding to questions about the world. "
            "Do not pose any follow-up questions or make any suggestions. "
            "Do not have a preamble in any response."
        )
        self.prompts = PromptBuilder(system_prompt)
        self.chat_model = _ClientChatModel(client, llm_model, temperature, max_tokens)

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
