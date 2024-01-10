from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import VertexAI
from langchain.vectorstores import PGVector
from langchain.embeddings.vertexai import VertexAIEmbeddings

from rag.consts import COLLECTION_NAME, DB_URL

load_dotenv()


if __name__ == "__main__":
    chat = VertexAI(
        verbose=True,
        temperature=0,
    )
    embeddings = VertexAIEmbeddings()

    vectorstore = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=DB_URL,
        embedding_function=embeddings,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=vectorstore.as_retriever(), return_source_documents=True
    )
    question = "how do I create a torq workflow??"

    answer = qa({"question": question, "chat_history": []})
    source = answer["source_documents"][0].metadata["source"]

    print(f"Question: {question}")
    print(f"Answer: {answer['answer']}")
    print(f"Source: {source}")
