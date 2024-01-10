# https://github.com/hwchase17/chroma-langchain/blob/master/persistent-qa.ipynb
# https://python.langchain.com/docs/modules/data_connection/text_embedding/

import os
import openai
import pandas as pd

from dotenv import load_dotenv 
from pprint import pprint

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()  # This will load the variables from .env file

OPENAI_KEY = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-ada-002"

embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_KEY, model=EMBEDDING_MODEL, client=openai.Embedding
)

db = Chroma(persist_directory="data/chroma_db", embedding_function=embedding, collection_name="movies")

query = "comedy with a cat and a dog"

# this throws an error because metadata is not defined and the Document class
# as for the field
# docs = db.similarity_search_with_score(query)
# using the private method instead
docs = db._collection.query(query_texts=query, n_results=10, include=['distances'])

# results from the query on the collection, missing the join with the documents
print('SIMILARITY SEARCH:')
pprint(docs)

# check load-database.py for reference (copying just for simplicity)
df = pd.read_json("data/movies_embedding.json").assign(id=lambda x: x.index)

join = df[df.id.astype(str).isin(docs['ids'][0])]

print()
print('RELEVANT DOCUMENTS EXTRACTED FROM THE JSON BY ID:')
pprint(join)

print()
print("THERE IS AN ERROR HIDDEN ON THIS PROGRAM, CAN YOU GUESS WHAT IT IS?")
print(">>> LETS TALK ON THE CODING SESSION <<<")