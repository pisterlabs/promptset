#%%
import weaviate
import os
from dotenv import load_dotenv


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(".env")

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE=os.environ.get("OPENAI_API_BASE" , "https://api.openai.com/v1")

auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)  # Replace w/ your Weaviate instance API key


weaviate_client = weaviate.Client(url=WEAVIATE_URL, 
auth_client_secret=auth_config,
additional_headers={
        # "X-Cohere-Api-Key": COHERE_API_KEY, # Replace with your cohere key
        "X-OpenAI-Api-Key": OPENAI_API_KEY, # Replace with your OpenAI key
        })


weaviate_client.schema.get()  # Get the schema to test connection

#%%

output = ""
with open("file1.csv", "rb") as f:
    output =  f.read().decode()

print(output)

#%%
weaviate_client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY))
embeddings = OpenAIEmbeddings()





#%%
# loader = TextLoader("../../modules/state_of_the_union.txt")
# documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 0,
    length_function = len,
    add_start_index = True,
)


documents = text_splitter.split_text(output)

print(documents)
vectorstore = Weaviate.from_texts(documents, embeddings, client=weaviate_client, by_text=False)


#%%

query = "utility"

# query_embedding = embeddings.embed_query(query)

docs = vectorstore.similarity_search(query , k=40)

print(docs)

docs = vectorstore.max_marginal_relevance_search(query , k=40 , fetch_k= 50,  lambda_mult= 0.9,)
print("*******************")
print(docs)

#%%
weaviate_client.schema.get()

#%%
result = weaviate_client.query.get("LangChain_1a9902e563d1449ebd85a09cd517ab51", ["text", ]).do()
print(result)
# %%

class_obj = {
    # Class definition
    "class": "JeopardyQuestion",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
          "model": "ada",
          "modelVersion": "002",
          "type": "text",
          "baseURL": OPENAI_API_BASE.replace("v1", "")
        }
      },
    # Property definitions
    "properties": [
        {
            "name": "category",
            "dataType": ["text"],
        },
        {
            "name": "question",
            "dataType": ["text"],
        },
        {
            "name": "answer",
            "dataType": ["text"],
        },
    ],

}
weaviate_client.schema.delete_class("JeopardyQuestion")
weaviate_client.schema.create_class(class_obj)

weaviate_client.schema.get()  # Get the schema to test connection

# %%

import pandas as pd

df = pd.read_csv("jeopardy_questions-350.csv", nrows = 100)
print(df)
# %%
from weaviate.util import generate_uuid5

with weaviate_client.batch(
    batch_size=200,  # Specify batch size
    num_workers=2,   # Parallelize the process
) as batch:
    for _, row in df.iterrows():
        question_object = {
            "category": row.category,
            "question": row.question,
            "answer": row.answer,
        }
        batch.add_data_object(
            question_object,
            class_name="JeopardyQuestion",
            uuid=generate_uuid5(question_object)
        )
# %%
weaviate_client.query.aggregate("JeopardyQuestion").with_meta_count().do()

#%%
import json

res = weaviate_client.query.get("JeopardyQuestion", ["question", "answer", "category"]).with_additional(["id"]).with_limit(2).do()

print(json.dumps(res, indent=4))
# %%
res = weaviate_client.query.get(
    "JeopardyQuestion",
    ["question", "answer", "category"])\
    .with_near_text({"concepts": "animals"})\
    .with_limit(5)\
    .do()

print(res)
# %%

import csv
# with open("file1.csv", "rb") as f:
with open("file1.csv") as f:
    output = f

    # output = output.decode()

    output = csv.DictReader(output , delimiter='\t')

    print(output)
    for i, row in enumerate(output):
        print(row)
        content = "\n".join(
            f"{k.strip()}: {v.strip()}"
            for k, v in row.items()
            # if k not in self.metadata_columns
        )


# %%

from forge.sdk.abilities.web.web_selenium import read_webpage

out = await read_webpage(None,"" , "https://en.wikipedia.org/wiki/Artificial_intelligence" ,"tech")


print(out)



# %%

from forge.sdk.abilities.web.web_search import web_search

out = await web_search(None,"" , "Latent Space podcast hosts Twitter handles")


print(out)

# %%
