from modules.vector_search import VectorSearch
import os
from langchain.embeddings.cohere import CohereEmbeddings
from utils.matching_engine import MatchingEngine
from dotenv import load_dotenv


class VectorStore:
    
    def __init__(self):
        BASEDIR = os.path.abspath(os.path.dirname("main.py"))
        load_dotenv(os.path.join(BASEDIR, '.env'))
        choere_api_key = os.getenv("COHERE_API_KEY")
        self.embedding_function = CohereEmbeddings(model="embed-multilingual-v2.0", cohere_api_key=choere_api_key)
        self.vectorsearch = VectorSearch()
        
    def get_vectorsearch(self):
        self.index_id, self.index_endpoint_id, self.project_id, self.region, self.vs_embedding_bucket = self.vectorsearch.create_index()

    def use_vector_search(self, doc_splits, blob_name):
        # add chunked data to the collection
        try:
            self.get_vectorsearch()
            print("Getting Vector Search Index")
            vs = MatchingEngine.from_components(
                project_id=self.project_id,
                region=self.region,
                gcs_bucket_name=f"gs://{self.vs_embedding_bucket}".split("/")[2],
                embedding=self.embedding_function,
                index_id=self.index_id,
                endpoint_id=self.index_endpoint_id,
            )
            texts = [doc.page_content for doc in doc_splits]
            metadatas = [
                [
                    {"namespace": "source", "allow_list": [doc.metadata["source"]]},
                    {"namespace": "document_name", "allow_list": [doc.metadata["document_name"]]},
                    {"namespace": "chunk", "allow_list": [str(doc.metadata["chunk"])]},
                ]
                for doc in doc_splits
            ]
            print (f"Adding {blob_name} to Vertex Search")
            try:
                doc_ids = vs.add_texts(texts=texts, metadatas=metadatas)
                print(f"Added the follwing Document ID's to Vertex Search {doc_ids}\n")
                
            except Exception as e:
                print(f"Error: {e}")
 
            print(f"Successfully added {blob_name} to Vertex Search\n")
        except Exception as e:
            print(f"Error: {e}")

    # Load the persistent vectordb
    def get_vectordb(self):
        #return Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_function)
        self.get_vectorsearch()
        return MatchingEngine.from_components(project_id=self.project_id, region=self.region, gcs_bucket_name=f"gs://{self.vs_embedding_bucket}".split("/")[2], embedding=self.embedding_function, index_id=self.index_id, endpoint_id=self.index_endpoint_id)