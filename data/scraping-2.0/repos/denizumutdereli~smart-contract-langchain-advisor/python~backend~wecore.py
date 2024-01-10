import os
from typing import Any

import weaviate
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.weaviate import Weaviate

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

client = weaviate.Client("http://localhost:8080")

def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()

    # Weaviate vector store
    docsearch = Weaviate(
        client=client,
        index_name="SmartContract", 
        text_key="text", 
        embedding=embeddings,
        attributes=[
            "text",
            "source",
        ],
        by_text=True,
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    response = qa({"query": query})
    result = response.get("result")
    source_documents = response.get("source_documents", [])

    # sources = [
    #     {"content": doc.page_content, "source": doc.metadata.get("source")}
    #     for doc in source_documents
    # ]

    contract_names = [
        os.path.basename(doc.metadata.get("source")) for doc in source_documents
    ]

    return {
        "query": query,
        "result": result,
        #'source_codes': sources,
        "source_contracts": contract_names,
    }


# if __name__ == "__main__":
#     print(run_llm(query="what is assembly in solidity?"))
