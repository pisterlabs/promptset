from langchain.document_loaders import ApifyDatasetLoader
from langchain.document_loaders.base import Document
# from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper
import apify_client
import os

# https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/milvus.html
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus

embeddings = OpenAIEmbeddings()

class ApifyTool:
    def __init__(self):
        self.apify = ApifyWrapper()
        self.indexes = {}
        self.client = apify_client.ApifyClient(token=os.environ.get("APIFY_API_TOKEN"))
        for dataset in self.client.datasets().list(unnamed=True, desc=True).items: # TODO may overflow page
            # Load existing datasets
            loader = ApifyDatasetLoader(
                dataset_id=dataset['id'],
                dataset_mapping_function=lambda dataset_item: Document(
                    page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
                ),
            )
            self._add_to_milvus(loader)
            # self._add_to_vectordb(loader)
        print("Loaded existing datasets.")
            
    def _add_to_milvus(self, loader):
        self.indexes[loader.dataset_id] = Milvus.from_documents(
            loader.load(), 
            embeddings, 
            connection_args={"host": "milvus", "port": "19530"}
        )
        
    # def _add_to_vectordb(self, loader):
    #     vector_db = VectorstoreIndexCreator().from_loaders([loader])
    #     self.indexes[loader.dataset_id] = vector_db
        
    def scrape_website(self, url: str):
        if url in self.indexes:
            return
        loader = self.apify.call_actor(
            actor_id="apify/website-content-crawler",
            run_input={"startUrls": [{"url": url}]},
            dataset_mapping_function=lambda item: Document(
                page_content=item["text"] or "", metadata={"source": item["url"]}
            ),
        )
        # Here, I want to dump it into Milvus instead of Vectorstore because duckdb won't compile
        self._add_to_milvus(loader)
        # self._add_to_vectordb(loader)
        return "Have successfully scraped the website {url} and stored the results in a dataset."
        
    def query(self, query):
        full_results = []
        for index in self.indexes.items():
            result = index.query_with_sources(query)
            if result:
                full_results.append(f"{result['answer']}\n\n{result['sources']}")
        if full_results:
            return "\n\n".join(full_results)
        else:
            return "No results found."