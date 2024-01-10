import logging
import os
from typing import Any, Dict, List, Optional, Type

import chromadb
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.storage import LocalFileStore
from langchain.vectorstores import Chroma


class RobustChroma(Chroma):
    """Handle UnicodeDecodeErrors and don't die, just skip them."""

    @classmethod
    def from_documents(
        cls: Type[Chroma],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "langchain",
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,  # Add this line
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a list of documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings
            collection_metadata (Optional[Dict]): Collection configurations.
                                                  Defaults to None.

        Returns:
            Chroma: Chroma vectorstore.
        """

        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            **kwargs,
        )

        last_file = None
        texts, metadatas = [], []
        for i, doc in enumerate(documents):
            filename = doc.metadata["source"]
            print(f"Adding {i}th document - {filename}")
            try:
                if filename != last_file:
                    if last_file:
                        chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                    texts, metadatas = [], []
                    last_file = filename

                # Build up a cache of documents to add...
                if RobustChroma.is_encodable(doc.page_content):
                    texts.append(doc.page_content)
                    metadatas.append(doc.metadata)

            # We should not be called...
            except UnicodeDecodeError:
                logging.warning(
                    f'Skipping document due to UnicodeDecodeError: {doc.metadata["source"]}'
                )
                continue  # Skip to the next document

        return chroma_collection

    @staticmethod
    def is_encodable(s):
        try:
            s.encode("utf-8")
        except UnicodeEncodeError:
            logging.warning(f"Skipping document due to UnicodeDecodeError: {s}")
            return False
        return True


def main():
    logging.getLogger("langchain").setLevel(logging.DEBUG)

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

    vectordb = RobustChroma.from_documents(
        motif_docs, embedding=cached_embedder, persist_directory="data"
    )
    vectordb.persist()

    # Setup a simple buffer memory system to submit with the API calls to provide prompt context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create a ConversationalRetrievalChain from the LLM, the vectorstore, and the memory system
    qa = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0.8),
        vectordb.as_retriever(),
        memory=memory,
        verbose=True,
    )

    # Ask some questions
    print(qa({"question": "What are the different types of network motif?"})["answer"])

    print(qa({"question": "What is an example of a biological network motif?"})["answer"])


if __name__ == "__main__":
    main()
