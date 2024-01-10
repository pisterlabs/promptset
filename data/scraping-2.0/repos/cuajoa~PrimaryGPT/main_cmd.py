# scritp que toma los embeddings de la base de dats vectordb y los usa para responder preguntas

from scripts.vectordb import vectordb
from langchain.chat_models import ChatOpenAI
from colorama import Fore
from langchain.chains import RetrievalQA

_vectordb = vectordb("fondoscollection")
_vectorRetriever=_vectordb.get_as_retriever()


llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=_vectorRetriever, chain_type="map_reduce"
)

while True:
    print(Fore.WHITE)
    query = input("> ")
    answer = qa_chain.run(query=query)

    print(Fore.GREEN, answer)
