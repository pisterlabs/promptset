from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage
from langchain.vectorstores import Milvus
from langchain.llms import OpenAI, OpenAIChat
import os


os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://172.29.7.155:8000/v1"
llm_completion = OpenAI(model_name="vicuna-13b-v1.5")
llm_chat = OpenAIChat(model_name="vicuna-13b-v1.5")

embedding = HuggingFaceEmbeddings()

db = Milvus(embedding_function=embedding, collection_name="arXiv_prompt",
            connection_args={"host": "172.29.4.47", "port": "19530"})

# openai兼容示例 以langchain为例
llm = ChatOpenAI(model_name="vicuna-13b-v1.5")
llm.predict_messages([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
