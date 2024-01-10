# currently all memories are handled by tools.py

from langchain import FAISS, OpenAIEmbeddings

def long_term_memory(docs):
        vectorstore = FAISS.from_texts([docs], embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()