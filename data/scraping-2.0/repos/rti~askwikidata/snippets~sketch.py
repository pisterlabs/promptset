from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
import pandas as pd
from annoy import AnnoyIndex
import torch

device = "cuda"
print("Loading models...")
# embedding_model_name = "BAAI/bge-small-en-v1.5"
# embedding_model_name = "BAAI/bge-base-en-v1.5"
embedding_model_name = "BAAI/bge-large-en-v1.5"
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
    query_instruction="Represent this sentence for searching relevant passages: ",
)

# cache_file = "cache-1280-0-BAAI-bge-small-en-v1.5.json"
# df = pd.read_json(cache_file)
# len(df)

chunk1 = """
Berlin inception on 1244.
Berlin time of earliest written record on 1237.
Berlin time of earliest written record on 1244.
Berlin is next to river or lake or sea Spree.
Berlin is next to river or lake or sea Großer Wannsee.
Berlin is next to river or lake or sea Lake Tegel.
Berlin is next to river or lake or sea Havel.
Berlin is next to river or lake or sea Dahme.
Berlin is next to river or lake or sea Müggelsee.
Berlin is next to river or lake or sea Aalemannkanal.
Berlin is next to river or lake or sea Neukölln Ship Canal.
Berlin is next to river or lake or sea Luisenstadt Canal.
Berlin is next to river or lake or sea Teltow Canal.
Berlin is next to river or lake or sea Landwehr Canal.
Berlin is next to river or lake or sea Westhafen Canal.
Berlin is next to river or lake or sea Gosen Canal.
Berlin is next to river or lake or sea Tegeler Fließ.
Berlin is next to river or lake or sea Berlin-Spandau Ship Canal.
Berlin had received award Princess of Asturias Award for Concord in 2009.
Berlin had received award City of Design in 2006.
Berlin had received award Führerstadt in 1937.
"""
chunk2 = """
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
No item label found: human settlement in Germany
No item label found country Germany.
No item label found is a city.
Mildura is next to river or lake or sea Murray River.
Melbourne has hashtag Melbourne.
Melbourne maintained by wikiproject WikiProject Melbourne.
Melbourne coat of arms coat of arms of Melbourne.
Schwerin is next to river or lake or sea Schweriner See.
Schwerin is next to river or lake or sea Burgsee.
Schwerin is next to river or lake or sea Fauler See.
Schwerin is next to river or lake or sea Grimkesee.
Schwerin is next to river or lake or sea Heidensee.
Schwerin is next to river or lake or sea Große Karausche.
Schwerin is next to river or lake or sea Lankower See.
Schwerin is next to river or lake or sea Medeweger See.
Schwerin is next to river or lake or sea Neumühler See.
Schwerin is next to river or lake or sea Ostorfer See.
Schwerin is next to river or lake or sea Pfaffenteich.
Schwerin is next to river or lake or sea Ziegelsee.
Schwerin is next to river or lake or sea Aubach.
Schwerin is next to river or lake or sea Stör.
Berlin owner of Altes Stadthaus, Berlin.
Berlin owner of Charité.
Berlin owner of Poststadion.
Berlin owner of Verkehrsverbund Berlin-Brandenburg.
Berlin owner of Bröhan Museum.
Berlin owner of Friedrich-Ludwig-Jahn-Sportpark.
Berlin owner of Deutschlandhalle.
Berlin owner of Funkturm Berlin.
Berlin owner of Stadion An der Alten Försterei.
Berlin owner of ResearchGate.
Berlin owner of BEHALA.
Berlin owner of Mommsenstadion.
Berlin owner of Flughafen Berlin Brandenburg GmbH.
Berlin owner of Olympiapark-Amateurstadion.
Berlin owner of Joachimstraße 6/8.
Berlin owner of Hans-Zoschke-Stadion.
Berlin owner of Stadium Buschallee.
Berlin owner of Wohnungsbaugesellschaft Berlin-Mitte.
Berlin owner of Berliner Stadtwerke.
"""
chunk3 = "through berlin runs river spree"
question = "Which River runs through Berlin?"
chunk_embed1 = embedding_model.embed_documents([chunk1])
chunk_embed2 = embedding_model.embed_documents([chunk2])
chunk_embed3 = embedding_model.embed_documents([chunk3])
# question_embed = embedding_model.embed_documents([question])
question_embed = embedding_model.embed_query(question)
print("Creating embedding index...")
embed_dims = len(chunk_embed1[0])
index = AnnoyIndex(embed_dims, "angular")
# index = AnnoyIndex(embed_dims, "dot")
index.add_item(0, chunk_embed1[0])
index.add_item(1, chunk_embed2[0])
index.add_item(2, chunk_embed3[0])
index.build(10)
# query_embed_float = [float(value) for value in question_embed[0]]
query_embed_float = [float(value) for value in question_embed]
nns = index.get_nns_by_vector(query_embed_float, 4, include_distances=True)
# index2 = AnnoyIndex(embed_dims, "dot")
# index2.add_item(0, chunk_embed1[0])
# index2.add_item(1, chunk_embed2[0])
# index2.add_item(2, chunk_embed3[0])
# index2.build(10)
# nns2 = index2.get_nns_by_vector(query_embed_float, 4, include_distances=True)
print(nns)
# print(nns2)

torch.Tensor(chunk_embed1[0]) @ torch.Tensor(question_embed)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings_1 = model.encode(chunk1, normalize_embeddings=True)
embeddings_2 = model.encode(question, normalize_embeddings=True)
# embeddings_2 = model.encode("Represent this sentence for searching relevant passages: " + question, normalize_embeddings=True)
embeddings_1 @ embeddings_2

torch.Tensor(embeddings_1) @ torch.Tensor(embeddings_2)


# In NLP, you might prefer using the dot product over cosine similarity in cases
# situations where the length of the vector carries meaningful information about
# the text, such as its importance or relevance in a certain context. For
# instance, in information retrieval or document ranking systems, where both the
# direction and magnitude of embedding vectors can contribute to the relevance of
# a document in relation to a query, the dot product might be more informative
# than cosine similarity. In such cases, the magnitude could represent the
# strength or confidence of the features encoded in the vector.


pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.width = 0

config = {
    "chunk_size": 1280,
    "chunk_overlap": 0,
    "index_trees": 1024,
    "retrieval_chunks": 5,
    "context_chunks": 5,
    "embedding_model_name": "BAAI/bge-large-en-v1.5",
    "reranker_model_name": "BAAI/bge-reranker-base",
    "qa_model_url": "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
}
askwikidata = AskWikidata(**config)
askwikidata.setup()
question = "Which River runs through Berlin?"
retrieved = askwikidata.retrieve(question)
print(askwikidata.context(retrieved))
retrieved


for _,i in askwikidata.df[askwikidata.df["text"].str.contains("Spree", na=False)].iterrows():
    print("===============")
    print(i["id"])

askwikidata.df.iloc[21]

text=askwikidata.df.iloc[21]["text"]
embedding_model.embed_documents([text])[0][0]
embedding_model.embed_documents([text])[0][1]

chunk_embed1[0][0]


len(askwikidata.df)
askwikidata.df[1]

for i, r in askwikidata.df.iterrows():
    if i == 10000:
        print(i)
        print(r["id"])
        print(r["text"])
        print(r["source"])
        print(r["embeddings"])
