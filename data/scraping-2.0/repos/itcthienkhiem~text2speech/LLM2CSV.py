from gradio_client import Client
from pydantic import BaseModel
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse

from langchain.llms import OpenAI
from langchain.embeddings.gpt4all import GPT4AllEmbeddings

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI

#OpenAIEmbeddings.api_base = 'https://api.nova-oss.com/v1'
#OpenAIEmbeddings.organization = "org-Q18CrNbEIghiXqzj3PI2tsC5"
#OpenAIEmbeddings.api_key = 'nv-v88q46u4Ha4Q6Qrb1bPRN0V4x0SSOvL3Ue3CvK9Wi8PqG8QM'
#OpenAIEmbeddings.model = 'gpt-4'
class Item(BaseModel):
    q: str

app = FastAPI()
#embeddings = OpenAIEmbeddings(openai_api_key="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")
#embeddings = OpenAIEmbeddings(openai_api_key="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")
embeddings = GPT4AllEmbeddings()

#llm = OpenAI(openai_api_key="nv-v88q46u4Ha4Q6Qrb1bPRN0V4x0SSOvL3Ue3CvK9Wi8PqG8QM",
#                 openai_api_base='https://api.nova-oss.com/v1', model_name='gpt-4', temperature=0)
llm = ChatOpenAI(openai_api_key="nv-v88q46u4Ha4Q6Qrb1bPRN0V4x0SSOvL3Ue3CvK9Wi8PqG8QM",
                openai_api_base='https://api.nova-oss.com/v1', model_name='gpt-4', temperature=0)

loader = CSVLoader(
        file_path="data/excel_file_example.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["P", "Q", "M"],
        },
    encoding='utf-8'
    )

data = loader.load()
#print(data)
@app.get("/getResult")
async def getResult(item: Item):
    #print('hih')
    print(item.q)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    red_alert_texts = text_splitter.split_documents(data)

    red_alert_db = Chroma.from_documents(red_alert_texts, embeddings)
    red_alert_retriever = red_alert_db.as_retriever()
    red_alert_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=red_alert_retriever)

    query = item.q+'"?'
    text = red_alert_qa.run(query)
    #print(text)
    return {text}


