"""
- Semantic Search goes beyond simple keyword matching
- Instead of exact keywords, SS can search for related concepts ("intent")
- Relations between two words are represented by cosine similarity
- Words can be converted to embedding (list of floats); embeddings of textual information relies on their semantics

- generating embeddings is a ML task, but pre-trained models are available (Openai Embeddings API)
- given a document, it spit out an embedding

***

Basic PoC pt 1
- PS -> LLM: Send initial prompt (fake queries)
- LLM -> PS: LLM generates an 'intention' (basically Openai Embeddings API) converts initial prompt (fake queries) into embeddings
- LLM -> PS: Return 'intention' (embeddings made from fake queries) to Python Service and Store embeddings in Chroma
- Chroma is VectorDB
- (Create a prompt where you provide an input, with prompt instruction to translate that human input into query description)
- PS: performs Top K retrieve in VectorDB
- (Take query description and run a Top-K search on previously made embeddings, observe whether query returned matched the 'intention')
- PS -> VDB: Retrieves most relevant fake query  based on intention
- VDB -> PS: Returns most relevant query description
- (Success is defined by fact that LLM can translate prompt into most correct query (intention) for the prompt; )

*** Proposed examples:

User prompt: 
- I would like to analyze the price movements of $FXS and understand the price trend, and whether or not it is bullish or bearish

> I would like to understand how FRAX peg mechanism works

> I would like to know which projects Frax Finance collaborates the most with

Prompt Chain should return: Top candidates for the task
- FXS exponential moving average <some description>
- FXS 1 day price change <some description>
- FXS 7 day price change <some description>

> FRAX AMO has a decollateralize function
> FRAX AMO has a recollateralize function

> FRAX holds this most XYZ token in its treasury
> FRAX uses this protocol as a swap facility
"""
import os
import json
import openai 
import chromadb

# use python-dotenv to get API key
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")


def get_embedding(text, model='text-embedding-ada-002'):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# indicate json file headlines are stored in.
with open('fxs.json', 'r') as file:
    token_description = json.load(file)

embeddings = []
for description in token_description:
    embedding = get_embedding(description['description'])
    embeddings.append(embedding)


# setup vector database
client = chromadb.Client()
collection = client.create_collection(name="token_description")


ids = [str(t['id']) for t in token_description] #chroma requires ids to be strings
documents = [t['description'] for t in token_description]
metadata = [{'name': t['name'], 'behavior': t['behavior']} for t in token_description]
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadata,
    embeddings=embeddings
)

query = "FXS holders are ambivalent."
embedding = get_embedding(query)
results = collection.query(query_embeddings=[embedding], n_results=1)

print(results)