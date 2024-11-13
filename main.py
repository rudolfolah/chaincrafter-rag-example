from chonkie import TokenChunker
from tokenizers import Tokenizer
from chaincrafter import Chain, Prompt
from chaincrafter.models import OpenAiChat
import openai
import faiss
import numpy as np

LLM_MODEL = 'llama-3.2-3b-instruct'
EMBEDDING_MODEL = 'text-embedding-nomic-embed-text-v1.5'

tokenizer = Tokenizer.from_pretrained("gpt2")

chunker = TokenChunker(tokenizer)

# Source: https://en.wikipedia.org/wiki/France
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
""".splitlines()

all_chunks = []
for doc in documents:
    chunks = chunker(doc)
    all_chunks.extend(chunks)

chunk_texts = [chunk.text for chunk in all_chunks]

def get_embeddings(texts):
    response = openai.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    embeddings = [data.embedding for data in response.data]
    return embeddings

chunk_embeddings = get_embeddings(chunk_texts)

embedding_dim = len(chunk_embeddings[0])
index = faiss.IndexFlatL2(embedding_dim)
chunk_embeddings_np = np.array(chunk_embeddings).astype('float32')
index.add(chunk_embeddings_np)

chat_model = OpenAiChat(temperature=0.55, model_name=LLM_MODEL)
system_prompt = Prompt("You are a helpful assistant, responding to questions about the world.")

queries = [
    "Tell me about French geography, specifically about things close to the border of Belgium. Write it as a TV show script.",
    "What is interesting about French royalty? Write a poem.",
    "What is the best French food? Mention at least one of the contextual facts.",
]

for query in queries:
    print(f"--- {query}")
    query_embedding = get_embeddings([query])[0]
    query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)

    k = 3
    D, I = index.search(query_embedding_np, k)
    retrieved_chunks = [chunk_texts[i] for i in I[0]]

    context = "\n\n".join(retrieved_chunks)
    user_prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"

    question_prompt = Prompt(user_prompt)
    chain = Chain(
        system_prompt,
        (question_prompt, "answer"),
    )
    messages = chain.run(chat_model)
    for message in messages:
        print(f"{message['role']}: {message['content']}")
    print("\n\n")
