
#from PyPDF2 import PdfReader
#from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain import FAISS
#from langchain.chains.question_answering import load_qa_chain
#from langchain.llms import OpenAI

import openai
from openai import OpenAI
import os
from core import msMLAIapp as msai
from core import config as apiSetting

keyAI = apiSetting.config["apiKey"]

openai.api_key = keyAI  
os.environ["OPENAI_API_KEY"] = keyAI
client = OpenAI()
def ask2Bot(questionList):
    params = dict(model="gpt-3.5-turbo", messages=questionList)
    response = client.completions.create(**params)
    return response

def msAsking(question):
     return msai.callBot(question)[1]
