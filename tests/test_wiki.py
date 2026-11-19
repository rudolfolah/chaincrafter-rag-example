from types import SimpleNamespace

import pytest

import wiki as wiki_module
from retriever import RetrievedChunk
from wiki import WikiRetrievalAugmentedGenerator, wiki_documents


class FakePrompts:
    def system_prompt(self):
        return object()

    def ingest_prompt(self, *args):
        return object()


class CapturingChain:
    inputs = None

    def __init__(self, *args):
        pass

    def run(self, model, inputs):
        type(self).inputs = inputs
        return [
            {
                "role": "assistant",
                "content": "---\ntitle: Generated\nsummary: Test\ncategory: General\ndate: 2026-07-17\n---\n# Generated",
            }
        ]


def make_wiki(vault_path):
    instance = WikiRetrievalAugmentedGenerator.__new__(
        WikiRetrievalAugmentedGenerator
    )
    instance.vault_path = vault_path
    return instance


def test_wiki_documents_uses_explicit_root_and_excludes_structural_files(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    page = nested / "Page.md"
    nested_schema = nested / "SCHEMA.md"
    nested_schema.write_text("# Nested schema page", encoding="utf-8")
    page.write_text("# Page", encoding="utf-8")
    (tmp_path / "SCHEMA.md").write_text("schema", encoding="utf-8")
    (tmp_path / "index.md").write_text("index", encoding="utf-8")

    assert wiki_documents(tmp_path) == [page, nested_schema]

    wiki = make_wiki(tmp_path)
    wiki._update_index(tmp_path)
    assert "[[nested/Page]]" in (tmp_path / "index.md").read_text(encoding="utf-8")



def test_answer_selects_page_with_dotted_name(monkeypatch, tmp_path):
    page = tmp_path / "release.v2.md"
    page.write_text("# Release v2\nSelected content", encoding="utf-8")
    wiki = make_wiki(tmp_path)
    wiki.prompts = SimpleNamespace(
        system_prompt=lambda: object(),
        select_pages_prompt=lambda *args: object(),
        wiki_answer_prompt=lambda *args: object(),
    )
    wiki.chat_model = object()
    wiki.retriever = SimpleNamespace(is_ready=False)

    class PageSelectingChain:
        calls = 0
        answer_inputs = None

        def __init__(self, *args):
            pass

        def run(self, model, inputs):
            type(self).calls += 1
            if type(self).calls == 1:
                return [{"role": "assistant", "content": "release.v2"}]
            type(self).answer_inputs = inputs
            return [{"role": "assistant", "content": "Answer"}]

    monkeypatch.setattr(wiki_module, "Chain", PageSelectingChain)

    wiki.answer("What changed?")

    assert "Selected content" in PageSelectingChain.answer_inputs["full_pages_content"]

def test_atomic_page_write_rejects_duplicate_normalized_targets(tmp_path):
    wiki = make_wiki(tmp_path)
    pages = [
        {"title": "A/B", "content": "# First"},
        {"title": "A-B", "content": "# Second"},
    ]

    with pytest.raises(ValueError, match="multiple generated pages"):
        wiki._write_pages_atomically(pages)

    assert list(tmp_path.glob("*.md")) == []


def test_page_write_rejects_collision_with_different_existing_title(tmp_path):
    existing = tmp_path / "A-B.md"
    existing.write_text("---\ntitle: Existing\n---\n# Existing", encoding="utf-8")
    wiki = make_wiki(tmp_path)

    with pytest.raises(ValueError, match="collides with existing file"):
        wiki._write_pages_atomically(
            [{"title": "A/B", "content": "---\ntitle: A/B\n---\n# A/B"}]
        )

    assert "Existing" in existing.read_text(encoding="utf-8")


def test_ingest_uses_retrieval_provenance_for_relevant_page(monkeypatch, tmp_path):
    source = tmp_path / "Source.md"
    source_content = "---\ntitle: Source\n---\n# Source\nOriginal page"
    source.write_text(source_content, encoding="utf-8")
    wiki = make_wiki(tmp_path)
    wiki.prompts = FakePrompts()
    wiki.chat_model = object()
    wiki.retriever = SimpleNamespace(is_ready=True, documents=[source])
    wiki.get_retrieved_documents = lambda query: [
        RetrievedChunk(
            text="generated context\n\nOriginal page",
            original_text="Original page",
            document_index=0,
            source=source,
        )
    ]
    wiki.prepare = lambda: wiki
    monkeypatch.setattr(wiki_module, "Chain", CapturingChain)

    wiki.ingest("new material")

    assert source_content in CapturingChain.inputs["relevant_pages_content"]
    assert (tmp_path / "Generated.md").exists()
