# from langchain.llms import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {"temperature": 0.5, "max_length": 64}
model_kwargs = {'device': 'cpu'}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

documents = [
    "The cat is on the mat.",
    "There is a cat on the mat.",
    "The dog is in the yard.",
    "There is a dog in the yard.",
]

doc_embeddings = hf.embed_documents(documents)