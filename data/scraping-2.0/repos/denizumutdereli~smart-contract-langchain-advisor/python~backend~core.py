import os
from typing import Any

import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from constants import PINECONE_INDEX

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX, embedding=embeddings
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=False,
    )

    return qa({"query": query})

# if __name__ == "__main__":
#     print(
#         run_llm(
#             query="""what's a inheritence example """
#         )
#     )
