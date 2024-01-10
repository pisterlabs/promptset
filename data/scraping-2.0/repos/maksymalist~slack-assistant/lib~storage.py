import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from vectorize import vectorize_document, get_db

if os.path.exists("./storage"):
    vector_store = get_db()
    retriever = vector_store.as_retriever()
else:
    path = "./docs/example.pdf"
    vectorize_document(path)
    vector_store = get_db()
    retriever = vector_store.as_retriever()


llm = ChatOpenAI(
    model_name='gpt-3.5-turbo', 
    )
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)