from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import os
import openai
os.environ['OPENAI_API_KEY'] = "write your openai api key here"
openai.api_key = os.environ['OPENAI_API_KEY']

embeddings = OpenAIEmbeddings()

db = FAISS.load_local("750-900_index", embeddings)

new_db = FAISS.load_local("1_750_index", embeddings)

db.merge_from(new_db)

folder_path = 'final_store'

db.save_local(folder_path)
