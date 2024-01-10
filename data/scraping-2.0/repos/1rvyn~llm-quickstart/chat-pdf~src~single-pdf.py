import os
import sys
from dotenv import load_dotenv

from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage

# Load the OPENAI_API_KEY from the environment
load_dotenv()
# Then use openai_api_key in your script where needed
def make_chain(version):
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature="0",
    )

    embedding = OpenAIEmbeddings()

    if version == "0":
        vector_store = Chroma(
            collection_name="june-2023-quickstartsimulator",
            embedding_function=embedding,
            persist_directory="src/data/chroma/1",
        )
    elif version == "2":
        vector_store = Chroma(
            collection_name="june-2023-quickstartsimulator-2",
            embedding_function=embedding,
            persist_directory="src/data/chroma/2",
        )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )


if __name__ == "__main__":
    print(f'All arguments received: {sys.argv}')  # This will print all arguments passed to your script
    load_dotenv()
    version = sys.argv[1]
    chain = make_chain(version)

    chat_history = []

    question = sys.stdin.read().strip()

    # Generate answer
    response = chain({"question": question, "chat_history": chat_history})

    # Retrieve answer
    answer = response["answer"]
    source = response["source_documents"]
    refrences = ""
    if source:
        page_numbers = set(doc.metadata['page_number'] for doc in source)
        page_numbers_str = ', '.join(str(pn) for pn in page_numbers)
        refrences += f"\nYou can read about this on page {page_numbers_str} on our quick-start guide."
    
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    # Print answer
    print(f': {answer}\nReferences: {refrences}\n')
