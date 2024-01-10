from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from utils.llm import process_llm_response
import argparse
from utils.constants import VECTOR_DB_DIR


def qa(query: str) -> None:
    embedding = OpenAIEmbeddings()

    vectordb = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embedding)

    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    llm_response = qa_chain(query)
    process_llm_response(llm_response)


if __name__ == "__main__":
    query = "How can I load data from S3 into Snowflake?"
    args = argparse.ArgumentParser()
    args.add_argument("--query", type=str, default=query)
    args = args.parse_args()
    qa(query=args.query)
