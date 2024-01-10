"""
Use CrateDB Vector Search with OpenAI embeddings.

As input data, the example uses the canonical `state_of_the_union.txt`.

Synopsis::

    # Install prerequisites.
    pip install -r requirements.txt

    # Start database.
    docker run --rm -it --publish=4200:4200 crate/crate:nightly

    # Configure: Set environment variables.
    # Correct OpenAI API key should be used. SQL connection string fits a local instance of CrateDB.
    export OPENAI_API_KEY="<API KEY>"
    export CRATEDB_CONNECTION_STRING="crate://crate@localhost/?schema=doc"

    # Run program.
    python vector_search.py
"""  # noqa: E501
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import CrateDBVectorSearch


def main():

    # Load the document, split it into chunks, embed each chunk,
    # and load it into the vector store.
    state_of_the_union_url = "https://github.com/langchain-ai/langchain/raw/v0.0.325/docs/docs/modules/state_of_the_union.txt"
    raw_documents = UnstructuredURLLoader(urls=[state_of_the_union_url]).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = CrateDBVectorSearch.from_documents(documents, OpenAIEmbeddings())

    # Invoke a query, and display the first result.
    query = "What did the president say about Ketanji Brown Jackson"
    docs = db.similarity_search(query)
    print(docs[0].page_content)


if __name__ == "__main__":
    main()
