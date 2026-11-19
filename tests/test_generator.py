from retrieval_augmented_generator import GeneratedAnswer, RetrievalAugmentedGenerator
from retriever import RetrievedChunk


class FakeChain:
    def __init__(self, *args):
        pass

    def run(self, model, inputs):
        return [
            {"role": "user", "content": inputs["query"]},
            {"role": "assistant", "content": f"Answer from {inputs['context']}"},
        ]


class FakePrompts:
    def system_prompt(self):
        return object()

    def user_prompt(self):
        return object()


class FakeRetriever:
    def retrieve(self, query, num_neighbors):
        return [
            RetrievedChunk(
                text="retrieved context",
                original_text="retrieved context",
                document_index=0,
            )
        ]


def test_answer_returns_messages_text_and_retrieval_metadata(monkeypatch, capsys):
    monkeypatch.setattr("retrieval_augmented_generator.Chain", FakeChain)
    generator = RetrievalAugmentedGenerator.__new__(RetrievalAugmentedGenerator)
    generator.retriever = FakeRetriever()
    generator.prompts = FakePrompts()
    generator.chat_model = object()
    generator._num_neighbors = 3
    generator._retrieved_documents = {}

    answer = generator.answer("question")

    assert isinstance(answer, GeneratedAnswer)
    assert answer.query == "question"
    assert answer.text == "Answer from retrieved context"
    assert answer.retrieved_chunks[0].original_text == "retrieved context"
    assert generator._retrieved_documents["question"] == list(answer.retrieved_chunks)
    assert capsys.readouterr().out == ""
