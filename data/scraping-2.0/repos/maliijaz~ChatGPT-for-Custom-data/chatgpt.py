import os
import sys
import constants
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import openai
from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = constants.APIKEY
query = sys.argv[1] 
loader = PyPDFLoader("C:/Users/Ali/Desktop/Assignments/NUST Code_of_Ethics.pdf")
index = VectorstoreIndexCreator().from_loaders([loader])

print (index.query(query, llm=ChatOpenAI()))