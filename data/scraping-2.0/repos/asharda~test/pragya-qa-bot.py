import os
import sys
from dotenv import load_dotenv
from typing import Any, Dict, List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
import pinecone

load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

INDEX_NAME = "pragya-teachus-bot"

def run_llm(
    docsearch,
    query: str,
    k: int,
    chat_history: Dict,
) -> Dict:

    chat = ChatOpenAI(verbose=False, temperature=0)
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_type='similarity', search_kwargs={'k': k}),
        return_source_documents=False,
    )
    answer = chain.run({"query": query, "chat_history": chat_history})
    return {"answer": answer}

if __name__ == "__main__":

    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    k = 5  # number of documents to retrieve

    # user's question text input widget
    yellow = "\033[0;33m"
    green = "\033[0;32m"
    white = "\033[0;39m"

    chat_history = {}
    print(f"{yellow}---------------------------------------------------------------------------------")
    print('Welcome to the DocBot. You are now ready to start interacting with your documents')
    print('---------------------------------------------------------------------------------')
    while True:
        query = input(f"{green}Prompt: ")
        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print('Exiting')
            sys.exit()
        if query == '':
            continue
        result = run_llm(
            docsearch, query, k, chat_history
        )
        print(f"{white}Answer: " + result["answer"])
        chat_history[query] = result["answer"]
