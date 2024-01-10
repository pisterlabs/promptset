import os
import pickle
from langchain.chains import RetrievalQA
from langchain import OpenAI
from dotenv import load_dotenv


load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")


def connect_to_db():
    with open("faiss_store_openai.pkl", "rb") as file:
        vstore = pickle.load(file)
    llm = OpenAI(temperature=0, openai_api_key=OPEN_API_KEY)
    model = RetrievalQA.from_llm(llm=llm, retriever=vstore.as_retriever())
    return model
