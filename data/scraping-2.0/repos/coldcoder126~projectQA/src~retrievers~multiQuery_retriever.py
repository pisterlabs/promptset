from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from src.vectorize.process import load_chroma

# 将一个问题变为多个问题，需要传一个llm
question = "What are the approaches to Task Decomposition?"
db = load_chroma()




