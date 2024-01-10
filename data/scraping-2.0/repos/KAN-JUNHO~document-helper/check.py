import os

import faiss
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores.faiss import FAISS


if __name__ == "__main__":
    embeddings = GPT4AllEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    new_vectorestore = FAISS.load_local(
        "faiss_index_react/GPT4AllEmbeddings", embeddings
    )

    query = "헨젤과 그레텔에 대해 말해줘"
    embeddings_vector = embeddings.embed_query(query)
    docs = new_vectorestore.similarity_search(query)
    docs_vector = new_vectorestore.similarity_search_by_vector(embeddings_vector)
    for i in range(0, 4):
        print(docs[i])
        print(docs_vector[i])
        print(embeddings_vector[i])
        print("---------------")

    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )
    qa = RetrievalQA.from_llm(
        llm=chat, retriever=new_vectorestore.as_retriever(), verbose=True
    )
    print(qa.run("헨젤과 그레텔에 대해 말해줘"))
