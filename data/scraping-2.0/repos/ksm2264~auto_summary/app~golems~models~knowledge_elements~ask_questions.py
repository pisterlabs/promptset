from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from app.golems.models.knowledge_elements.templates import answerer_template

embeddings = OpenAIEmbeddings()
faiss_db = FAISS.load_local("faiss_index", embeddings)

query = "is retinal motion driven by behavior?"

retriever = faiss_db.as_retriever(search_kwargs=dict(k=20))
memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key='details')

llm = ChatOpenAI(model_name = 'gpt-3.5-turbo')
answerer = LLMChain(llm = llm, prompt = answerer_template, memory = memory)

while True:
    query = input('Question: ')

    answer = answerer.predict(
        question = query
    )

    print(answer)