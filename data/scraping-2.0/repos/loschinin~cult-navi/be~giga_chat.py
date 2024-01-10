from langchain.retrievers import WikipediaRetriever
from langchain.chat_models.gigachat import GigaChat
from langchain.chains import ConversationalRetrievalChain

retriever = WikipediaRetriever(lang='ru', load_max_docs=2)

giga = GigaChat(credentials='YWEyOWY4Y2EtYWI0MC00M2RlLWEzZDQtY2VkZTUwYzdhYTFhOjU4ODUwNTkxLTkzMjAtNGNiOC05YWZlLTQ1YjFjMmY3ODdiMw==', verify_ssl_certs=False)

docs = retriever.get_relevant_documents(query="Музей современного искусства Эрарта", lang="ru")

qa = ConversationalRetrievalChain.from_llm(giga, retriever=retriever)

questions = [
    "Дата основания Музей современного искусства Эрарта?",
]
chat_history = []

def llm_answer(question:str, chat_history:list, qa) -> str:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"])) 
    return result['answer']

print(llm_answer("Дата основания Музей современного искусства Эрарта?", chat_history, qa))