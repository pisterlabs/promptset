import logging
import os
import subprocess

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.storage import LocalFileStore
from langchain.vectorstores import OpenSearchVectorSearch

logging.getLogger("langchain").setLevel(logging.DEBUG)


def main():
    # Dropbox folder with academic papers
    PAPER_FOLDER = f"{os.getcwd()}/data/Network_Motifs/"
    paper_count = len(os.listdir(PAPER_FOLDER))
    print(f"You have {paper_count:,} Network Motif PDFs in `{PAPER_FOLDER}`.")

    # Set in my ~/.zshrc
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Load all PDFs from academic paper folder
    loader = PyPDFDirectoryLoader(PAPER_FOLDER, silent_errors=True)
    docs = loader.load()
    print(f"You have {len(docs)} document segments in `{PAPER_FOLDER}`.")

    # How many papers on network motifs?
    motif_docs = [doc for doc in docs if "motif" in doc.page_content]
    motif_doc_count = len(motif_docs)
    paper_count = len(set(doc.metadata["source"] for doc in motif_docs))
    print(
        f"You have {paper_count} papers mentioning network motifs split across {motif_doc_count} document segments in `{PAPER_FOLDER}`."
    )

    # Embed them with OpenAI ada model and store them in OpenSearch
    embeddings = OpenAIEmbeddings()
    fs = LocalFileStore("./data/embedding_cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, fs, namespace=embeddings.model
    )

    # Setup a OpenSearch to store the embeddings
    opensearch = OpenSearchVectorSearch(
        index_name="academic_papers",
        embedding_function=cached_embedder,
        opensearch_url="http://admin:admin@localhost:9200",
    )
    opensearch.add_documents(motif_docs, bulk_size=1024, verbose=True)

    # Setup a simple buffer memory system to submit with the API calls to provide prompt context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create a ConversationalRetrievalChain from the LLM, the vectorstore, and the memory system
    qa = ConversationalRetrievalChain.from_llm(
        OpenAI(model="gpt-3.5", temperature=0.8),
        opensearch.as_retriever(),
        memory=memory,
        verbose=True,
    )

    # Ask some questions...
    print(qa({"question": "What are the different types of network motif?"})["answer"])

    print(qa({"question": "What is an example of a biological network motif?"})["answer"])


if __name__ == "__main__":
    main()
