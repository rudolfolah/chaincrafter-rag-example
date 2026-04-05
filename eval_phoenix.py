import os

from retriever import ChunkerKind
from retrieval_augmented_generator import RetrievalAugmentedGenerator

import phoenix as px
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.evals import LLM
from phoenix.evals.metrics import DocumentRelevanceEvaluator
from phoenix.otel import register

px.launch_app()
tracer_provider = register(project_name="chaincrafter-rag-eval")
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


# Sources:
# - https://en.wikipedia.org/wiki/France
# - https://en.wikipedia.org/wiki/England
documents = """France, officially the French Republic, is a country located primarily in Western Europe.
Its overseas regions and territories include French Guiana in South America, Saint Pierre and Miquelon in the North Atlantic, the French West Indies, and many islands in Oceania and the Indian Ocean, giving it one of the largest discontiguous exclusive economic zones in the world.
Metropolitan France shares borders with Belgium and Luxembourg to the north, Germany to the northeast, Switzerland to the east, Italy and Monaco to the southeast, Andorra and Spain to the south, and a maritime border with the United Kingdom to the northwest.
Its metropolitan area extends from the Rhine to the Atlantic Ocean and from the Mediterranean Sea to the English Channel and the North Sea.
Its eighteen integral regions (five of which are overseas) span a combined area of 643,801 km2 (248,573 sq mi) and have a total population of 68.4 million as of January 2024.
France is a semi-presidential republic with its capital in Paris, the country's largest city and main cultural and commercial centre.
The Carolingian dynasty ruled France until 987, when Hugh Capet was crowned king of the Franks.
His descendants unified the country through wars and inheritance. From 1190, the Capetian rulers began to be referred as "kings of France" rather than "kings of the Franks".
Later kings expanded their directly possessed domaine royal to cover over half of modern France by the 15th century.
Royal authority became more assertive, centred on a hierarchically conceived society distinguishing nobility, clergy, and commoners.
The vast majority of France's territory and population is situated in Western Europe and is called Metropolitan France.
It is bordered by the North Sea in the north, the English Channel in the northwest, the Atlantic Ocean in the west and the Mediterranean Sea in the southeast.
Its land borders consist of Belgium and Luxembourg in the northeast, Germany and Switzerland in the east, Italy and Monaco in the southeast, and Andorra and Spain in the south and southwest.
Except for the northeast, most of France's land borders are roughly delineated by natural boundaries and geographic features: to the south and southeast, the Pyrenees and the Alps and the Jura, respectively, and to the east, the Rhine river.
Metropolitan France includes various coastal islands, of which the largest is Corsica.
Metropolitan France is situated mostly between latitudes 41° and 51° N, and longitudes 6° W and 10° E, on the western edge of Europe, and thus lies within the northern temperate zone.
Its continental part covers about 1000 km from north to south and from east to west.
Different regions have different styles. In the north, butter and cream are common ingredients, whereas olive oil is more commonly used in the south.[373] Each region of France has traditional specialties: cassoulet in the southwest, choucroute in Alsace, quiche in the Lorraine region, beef bourguignon in Burgundy, Provençal tapenade, etc. France is most famous for its wines[374] and cheeses, which are often named for the territory where they are produced (AOC).[375][376] A meal typically consists of three courses, entrée ('starter'), plat principal ('main course'), and fromage ('cheese') or dessert, sometimes with a salad served before the cheese or dessert.
French cuisine is also regarded as a key element of the quality of life and the attractiveness of France. A French publication, the Michelin Guide, awards Michelin stars for excellence to a select few establishments.[377][378] The acquisition or loss of a star can have dramatic effects on the success of a restaurant. By 2006, the Michelin Guide had awarded 620 stars to French restaurants.[379]
In addition to its wine tradition, France is also a major producer of beer and rum. The three main French brewing regions are Alsace (60% of national production), Nord-Pas-de-Calais, and Lorraine.
England is a country that is part of the United Kingdom. It is located on the island of Great Britain, of which it covers about 62%, and more than 100 smaller adjacent islands. England shares a land border with Scotland to the north and another land border with Wales to the west, and is surrounded by the North Sea to the east, the English Channel to the south, the Celtic Sea to the south-west, and the Irish Sea to the west. Continental Europe lies to the south-east, and Ireland to the west.
Since the early modern period the food of England has historically been characterised by its simplicity of approach and a reliance on the high quality of natural produce. During the Middle Ages and the Renaissance, English cuisine enjoyed an excellent reputation, though a decline began during the Industrial Revolution with increasing urbanisation. The cuisine of England has, however, recently undergone a revival, which has been recognised by food critics with some good ratings in Restaurant's best restaurant in the world charts.
Traditional examples of English food include the Sunday roast, featuring a roasted joint (usually beef, lamb, chicken or pork) served with assorted vegetables, Yorkshire pudding and gravy.
Other prominent meals include fish and chips and the full English breakfast (generally consisting of bacon, sausages, grilled tomatoes, fried bread, black pudding, baked beans, mushrooms and eggs).
Various [English] meat pies are consumed, such as steak and kidney pie, steak and ale pie, cottage pie, pork pie (usually eaten cold)[270] and the Cornish pasty.
[In England,] Sausages are commonly eaten, either as bangers and mash or toad in the hole. Lancashire hotpot is a well-known stew originating in the northwest.
[In England,] Some of the more popular cheeses are Cheddar, Red Leicester, Wensleydale, Double Gloucester and Blue Stilton
Traditional English dessert dishes include apple pie or other fruit pies; spotted dick – all generally served with custard; and, more recently, sticky toffee pudding. Sweet pastries include scones served with jam or cream, dried fruit loaves, Eccles cakes and mince pies as well as sweet or spiced biscuits.
""".splitlines()

queries = [
    "Tell me about French geography, specifically about things close to the border of Belgium. Write it as a TV show script.",
    # "What is interesting about French royalty? Write a poem.",
    # "What is interesting about English royalty? Write a poem.",
    # "What are the best French and English foods? Mention all of the contextual facts and compare them.",
]

# Workaround needed for unstructured
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

llm = LLM(provider="openai", model="gpt-4o")
relevance_eval = DocumentRelevanceEvaluator(llm=llm)

rag = RetrievalAugmentedGenerator(
    documents=documents,
    queries=queries,
    embedding_model="text-embedding-nomic-embed-text-v1.5",
    llm_model="google/gemma-3-4b",
    temperature=0.7,
    num_neighbors=3,
    chunker=ChunkerKind.DOCLING,
)
rag.run()

for query in queries:
    retrieved_documents = rag.get_retrieved_documents(query)

    print(f"\nQuery: {query}")
    for i, doc in enumerate(retrieved_documents):
        scores = relevance_eval.evaluate(
            {
                "input": query,
                "document_text": doc,
            }
        )
        print(f"Document {i + 1}: {scores[0].label} ({scores[0].score})")

input("Press enter to stop the Phoenix server")