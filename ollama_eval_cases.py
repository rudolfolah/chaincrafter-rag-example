"""Small hand-labeled retrieval cases using the actual feature-demo corpora.

Labels are deliberately small, not exhaustive relevance judgments. Each phrase
is a verbatim span of supporting original text, chosen to stay within a sentence
so Unstructured element boundaries do not require a whole paragraph to match.
A hit is ANY phrase after case-insensitive, whitespace-folded normalization of
both the phrase and the original chunk; generated contextual text is not evidence.
"""

from dataclasses import dataclass

from demo_contextual_retrieval import demo_documents
from main import documents, queries


@dataclass(frozen=True)
class Judgment:
    query: str
    relevant_phrases: tuple[str, ...]


@dataclass(frozen=True)
class DemoCase:
    name: str
    documents: tuple[str, ...]
    judgments: tuple[Judgment, ...]
    top_k: int


_FRENCH_BORDER_PHRASES = (
    "Metropolitan France shares borders with Belgium and Luxembourg",
    "Its land borders consist of Belgium and Luxembourg",
)


DEMO_CASES: dict[str, DemoCase] = {
    "contextual": DemoCase(
        name="contextual",
        documents=tuple(demo_documents),
        judgments=(
            Judgment(queries[0], (_FRENCH_BORDER_PHRASES[0],)),
            Judgment(
                "What is the capital of France?",
                ("capital in Paris",),
            ),
            Judgment(
                "Which country borders England to the north?",
                ("land border with Scotland to the north",),
            ),
        ),
        top_k=1,
    ),
    "hybrid": DemoCase(
        name="hybrid",
        documents=tuple(documents),
        judgments=(
            Judgment(
                queries[3],
                (
                    "butter and cream are common ingredients",
                    "olive oil is more commonly used in the south",
                    "cassoulet in the southwest",
                    "France is most famous for its wines",
                    "A meal typically consists of three courses",
                    "French cuisine is also regarded as a key element",
                    "awards Michelin stars for excellence",
                    "acquisition or loss of a star",
                    "Michelin Guide had awarded 620 stars",
                    "major producer of beer and rum",
                    "three main French brewing regions",
                    "food of England has historically been characterised",
                    "English cuisine enjoyed an excellent reputation",
                    "cuisine of England has, however, recently undergone a revival",
                    "Sunday roast",
                    "fish and chips",
                    "Various [English] meat pies",
                    "bangers and mash",
                    "Lancashire hotpot",
                    "Cheddar, Red Leicester",
                    "Traditional English dessert dishes",
                    "Sweet pastries include scones",
                ),
            ),
            Judgment(
                "Which cheeses are popular in England?",
                (
                    "Cheddar, Red Leicester, Wensleydale, Double Gloucester and Blue Stilton",
                ),
            ),
            Judgment(
                "Which guide awards stars to French restaurants?",
                (
                    "Michelin Guide, awards Michelin stars",
                    "Michelin Guide had awarded 620 stars",
                ),
            ),
        ),
        top_k=1,
    ),
    "reranking": DemoCase(
        name="reranking",
        documents=tuple(documents),
        judgments=(
            Judgment(queries[0], _FRENCH_BORDER_PHRASES),
            Judgment(
                "Which natural features mark France's land borders?",
                (
                    "the Pyrenees and the Alps and the Jura",
                    "the Rhine river",
                ),
            ),
            Judgment(
                "What is in a full English breakfast?",
                ("full English breakfast",),
            ),
        ),
        top_k=2,
    ),
}
