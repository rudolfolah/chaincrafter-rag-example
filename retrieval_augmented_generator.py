from chaincrafter import Chain
from chaincrafter.models import OpenAiChat

from prompt_builder import PromptBuilder
from retriever import Retriever, ChunkerKind

from trulens.core.otel.instrument import instrument as trulens_instrument
from trulens.otel.semconv.trace import SpanAttributes


class RetrievalAugmentedGenerator:
    def __init__(
        self,
        documents,
        queries,
        embedding_model,
        llm_model,
        num_neighbors=3,
        temperature=0.55,
        system_prompt=None,
        chunker: ChunkerKind = ChunkerKind.UNSTRUCTURED,
    ):
        self.queries = queries
        self.retriever = Retriever(documents, embedding_model, chunker=chunker)
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

    @trulens_instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def get_retrieved_documents(self, query):
        chunks = self.retriever.retrieve(query, num_neighbors=self._num_neighbors)
        self._retrieved_documents[query] = chunks
        return chunks

    @trulens_instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def _answer(self, query):
        chunks = self.get_retrieved_documents(query)
        chain = Chain(
            self.prompts.system_prompt(),
            (self.prompts.user_prompt(query, chunks), "answer")
        )
        return chain.run(self.chat_model)
