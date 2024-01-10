#!/usr/bin/env python
import logging
from langchain.llms import GPT4All
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.prompts import PromptTemplate
import workaround
from langchain.vectorstores.pgvector import PGVector
from langchain.chains.question_answering import load_qa_chain


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    logging.debug("Loading model ...")
    with workaround.suppress_stdout_stderr():
        llm = GPT4All(
            model="./models/llama-2-7b-chat.ggmlv3.q4_0.bin",
        )
    logging.debug("Loaded for CPU.")
    # Prepare MultiQueryRetriever
    # PGVector needs the connection string to the database.
    CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"  # NOSONAR (Disable SonarLint warning)

    # create embeddings
    embeddings = GPT4AllEmbeddings()
    store = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

    # Prepare query
    prompt_template = PromptTemplate.from_template(
        "Tell me about {content}. Do not exceed 42 tokens."
    )
    question = prompt_template.format(content="Hello World!")
    logging.debug("Start similarity search ...")
    docs = store.similarity_search(question)
    logging.debug(f"Results: {docs}")

    # Chain
    logging.debug("Start chain ...")
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=question)

    logging.debug(f"Response: {response}")


if __name__ == "__main__":
    main()
