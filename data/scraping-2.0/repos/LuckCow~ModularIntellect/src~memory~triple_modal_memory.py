from typing import Dict, List, Any
import time

from langchain.schema import BaseMemory
from langchain import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.memory.neo4j_docstore import Neo4jDocstore
from langchain.embeddings import OpenAIEmbeddings
from neo4j import GraphDatabase
import faiss

from src.utils.mapping_directory_loader import MappingDirectoryLoader


class TripleModalMemory:
    """Provides interface for storing and retrieving data from the Neo4j+FAISS database."""
    def __init__(self, uri, user, password):
        # TODO: load in existing documents into the faiss index, enforcing them to be in sync
        # or save the faiss index to disk (but then we need to either save every time something happens or risk corruption
        # setup the docstore and vector store
        neo4j_docstore = Neo4jDocstore(uri, auth=(user, password))
        embedding = OpenAIEmbeddings()
        index = faiss.IndexFlatL2(1536)
        faiss_vector_store = FAISS(embedding.embed_query, index, docstore=neo4j_docstore, index_to_docstore_id={})
        self.vector_store = faiss_vector_store

        # setup the memory interface
        class TripleModalAgentMemory(BaseMemory):
            """Langchain Memory class for chain interactions to be persisted into the Neo4j database."""
            memory_key: str = "history"

            def clear(self):
                # Clear the Neo4j graph
                with driver.session() as session:
                    query = "MATCH (n) DETACH DELETE n"
                    session.run(query)

            @property
            def memory_variables(self) -> List[str]:
                return [self.memory_key]

            def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
                text = inputs[list(inputs.keys())[0]]
                similar_interactions = self._retrieve_similar_interactions(text)
                # interactions_text = "\n".join(
                #     [f"{item['i.text']} ({item['i.timestamp']})" for item in similar_interactions])
                return {self.memory_key: similar_interactions}

            def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
                input = inputs[list(inputs.keys())[0]]
                output = outputs[list(outputs.keys())[0]]
                timestamp = inputs.get("timestamp", None)  # You can include an optional timestamp in the inputs
                if timestamp is None:
                    import datetime
                    timestamp = datetime.datetime.utcnow().isoformat()

                # TODO: identify and pass interaction identifiers
                interaction_id = '1'
                faiss_vector_store.add_texts([input, output],
                                             metadatas=[{'timestamp': timestamp, 'source': 'user',
                                                         'node_type': 'interaction', 'conversation_id': interaction_id},
                                                        {'timestamp': timestamp, 'source': 'agent',
                                                         'node_type': 'interaction', 'conversation_id': interaction_id}])

            def _store_interaction(self, text, timestamp, parent_id=None):
                # Add the input to the FAISS index
                faiss_vector_store.add_texts([text], metadatas=[{'timestamp': timestamp}])  # TODO: parent?

            def _retrieve_similar_interactions(self, text, k=5):
                # Search the FAISS index for similar interactions
                return faiss_vector_store.similarity_search(text, k)

        self.memory = TripleModalAgentMemory()

    def store_task(self, task: str, timestamp: str):
        """Store a task in the memory."""
        self.vector_store.add_texts([task], metadatas=[{'timestamp': timestamp, 'node_type': 'task'}])

    def store_notes(self, notes: str, sources, timestamp: str):
        """Store notes that come from existing memory."""
        self.vector_store.add_texts([notes], metadatas=[{'timestamp': timestamp,
                                                          'sources': sources,  # TODO: setup sources
                                                          'node_type': 'notes'}])

    def ingest_docs(self, path: str, chunk_size=1000, chunk_overlap=200):
        """Read, split and store code and other files from within the repository/folder."""
        loader = MappingDirectoryLoader(path, recursive=True, silent_errors=False)
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        documents = text_splitter.split_documents(raw_documents)
        for d in documents:
            d.metadata["chunk_size"] = chunk_size
            d.metadata["chunk_overlap"] = chunk_overlap
            d.metadata["node_type"] = "document"

        # Add documents to vector store in chunks with wait time to avoid getting rate limited
        # by the OpenAI API (20 requests per minute)
        for i in range(0, len(documents), 20):
            self.vector_store.add_documents(documents[i:i+20])
            print(f"Added documents {i} to {i+20}/{len(documents)} to vector store")
            time.sleep(60)


        #self.vector_store.add_documents(documents)

    def save(self):
        self.vector_store.save_local('../data', 'triple_modal_memory')

    def load(self):
        embedding = OpenAIEmbeddings()
        self.vector_store = FAISS.load_local('../data', embedding, 'triple_modal_memory')

    def verify(self):
        print(self.vector_store.index.ntotal)

    def search(self, query, k):
        return self.vector_store.similarity_search(query, k)


def test_ingest_save(mem):
    knowledge_path = r'C:\Users\colli\PycharmProjects\ModularIntellect\data\test_knowledgebase'
    # storage_path = '../data/langchain.pkl'
    mem.ingest_docs(knowledge_path)

    mem.save()


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv

    load_dotenv()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    mem = TripleModalMemory(uri, user, password)

    #test_ingest_save(mem)
    mem.load()
    mem.verify()

    print(mem.vector_store.similarity_search("What are the implementations of BaseChainLangAgent?", 3))

    # import datetime
    # timestamp = datetime.datetime.utcnow().isoformat()
    # #faiss_vector_store.add_texts(['Start of document ', 'middle of document', 'end of document'], metadatas=[{'timestamp': timestamp}, {'timestamp': timestamp}, {'timestamp': timestamp}])
    #
    # mem.save_context({"text": "The fictional nation of Jietao has 4 major cities: Niuy, Bietao, Cholkja, and Fosst"}, {})
    # mem.save_context({"text": "Leeroy is a blacksmith that lives in Niuy"}, {})
    # mem.save_context({"text": "Jaon is a mason that lives in Cholkja"}, {})
    #
    # print(mem.retrieve_similar_interactions("What nation does Leeroy live in?", 3))

    #driver.close()

