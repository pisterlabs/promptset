import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
from dotenv import load_dotenv
load_dotenv("/Users/spinokiem/Documents/Spino_DS_prj/building_a_chatbot")


# -------loading document-------
nqld = pd.read_excel("../../data/interim/nqld.xlsx")
rename_dict = {
    'h1': 'heading 1',
    'h2': 'heading 2',
    'h3': 'heading 3'
}
nqld.rename(columns=rename_dict, inplace=True)
nqld.fillna("", inplace=True)
nqld['document'] = 'HR.03.V3.2023. Nội quy Lao động'

# use DataFrameLoader of langchain to create a `langchain.schema.document.Document` object
loader = DataFrameLoader(nqld)
documents = loader.load()
# type(documents[0])


# -------setting up the embedder & vectordatabase-------
embedder = OpenAIEmbeddings()
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment='gcp-starter' # environment's name can be found at your acc at https://app.pinecone.io/
)

index_name = "itl-knl-base"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536
    )

# -------testing-------
# docsearch = Pinecone.from_documents(documents, embedder, index_name=index_name)

# # pinecone.delete_index("itl-knl-base") # delete all
# knl_base = pinecone.Index('itl-knl-base')
# knl_base.describe_index_stats()
