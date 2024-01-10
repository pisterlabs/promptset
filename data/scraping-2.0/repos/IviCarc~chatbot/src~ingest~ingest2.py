from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

import os
import openai


def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="1",
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        # verbose=True
    )

    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name="april-2023-economic",
        embedding_function=embedding,
        persist_directory="src/data/chroma",
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        # verbose=True,
    )


if __name__ == "__main__":
    load_dotenv()

    with open('prompt2.txt', 'r', encoding='utf-8') as file:
        # Lee todo el contenido del archivo
        prompt = file.read()

    # Imprime el contenido del archivo
    # print(prompt)

    chain = make_chain()
    chat_history = [
        SystemMessage(content=prompt)
    ]

    while True:
        print()
        question = input("Question: ")

        # Generate answer
        response = chain({"question": question, "chat_history": chat_history})

        # Retrieve answer
        answer = response["answer"]
        source = response["source_documents"]
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        # Display answer
        print("\n\nSources:\n")
        # for document in source:
            # print(f"Page: {document.metadata['page_number']}")
            # print(f"Text chunk: {document.page_content[:160]}...\n")
        print(f"Answer: {answer}")
