from langchain.vectorstores import Pinecone
import pinecone
from model_handler import OPENAI_API_KEY
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


index_name = 'v1-index-pinecone'
text_field = "text"
PIN_API_KEY = "681591b9-096b-4da3-8df2-9de5f89fba34"
ENV = "northamerica-northeast1-gcp"


index = pinecone.Index(index_name)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(api_key=PIN_API_KEY, environment=ENV)
vectorstore = Pinecone(index, embeddings.embed_query, text_field)








