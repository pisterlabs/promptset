import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import models
from sentence_transformers import SentenceTransformer
import pickle

from langchain.llms import OpenAI

from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Load Data & Prepare data
def load_data(data):
    return pd.read_csv(data)

def prepare_data(df):
    docx = df["Lawyer Names"].tolist()
    payload = df[["Rating","Jurisdiction","Charges","Days_of_disposal","City","Years_of_Experience","Type_of_Lawyer","Languages","Pro_bono","Lawyer Names","Lawfirms","Demography","Gender"]].to_dict("records")
    return docx, payload

def save_vectors(vectors):
    with open("vectorized_lawyers.pickle", "wb") as f:
        pickle.dump(vectors, f)
        
def load_vectors(vector_file):
    with open(vector_file, "rb") as f:
        my_object = pickle.load(f)
    return my_object
        


# Create a VectorDB client

client = QdrantClient(path="vector_database.db")
client.recreate_collection(collection_name="lawyers_collection",
                           vectors_config=models.VectorParams(
                               size=384, distance=models.Distance.COSINE
                           ))

# Vectorized our Data: Create Word Embeddings

model = SentenceTransformer('all-MiniLM-L6-v2')

# model = OpenAI(openai_api_key="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")

df = load_data(r"C:\Users\a21ma\OneDrive\Desktop\Datahack\DataHack_2_Tensionflow\Vector Database\recproject\FINALFINALFINALdataset.csv")
docx, payload = prepare_data(df)
vectors = model.encode(docx, show_progress_bar=True)
save_vectors(vectors)




text_to_predict = "Two brothers were tenant of a landlord in a commercial property. One brother had one son and a daughter (both minor) when he got divorced with his wife. The children went into the mother's custody at the time of divorce and after some years the husband (co tenant) also died. Now can the children of the deceased brother (co tenant) claim the right."

prompt = ChatPromptTemplate.from_template(
    "What could be the catagory for the query {input}, The Categories are [Banking and Finance, Civil, Constitutional, Consumer Protection,Corporate, Criminal, Environmental, Family,Human Rights, Immigration, Intellectual Property,Labor, Media and Entertainment, Medical,Real Estate, Tax] return 3 catagories in a array"
)

str_chain = prompt | model | StrOutputParser()

catagories = [i.replace("and ", "") for i in str_chain.invoke({"input": text_to_predict}).strip().replace("[",'').split(', ')]

# Store in VectorDB collection

client.upload_collection(collection_name="lawyers_collection",
                            payload=payload,
                            vectors=vectors,
                            ids=None,
                            batch_size=256)

vectorized_text = model.encode("Human").tolist()
results = client.search(collection_name="lawyers_collection",
                        query_vector=vectorized_text,
                        limit=10),

print(results)