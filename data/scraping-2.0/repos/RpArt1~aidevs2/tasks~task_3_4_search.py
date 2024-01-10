from tasks.abstractTask import AbstractTask
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.embeddings.openai import OpenAIEmbeddings
import logging
from utils.assigment_utils import AssigmentUtils
from uuid import uuid4



class SearchTask(AbstractTask):
    
    URL = "https://unknow.news/archiwum.json"
    QDRANT_COLLECTION = "unknow_news"
    
    def __init__(self, task_name, send_to_aidevs, mock):
        super().__init__(task_name, send_to_aidevs, mock)
        # set up qdrant client
        self.qdrant_client = QdrantClient("localhost", port=6333, timeout=1000)

    
    def solve_task(self):
        super().solve_task()
        
    def process_task_details(self):

        self.import_from_url()
        question = self.assignment_body['question']
        question_vector =  OpenAIEmbeddings().embed_query(question)
        search_result = self.qdrant_client.search(
            collection_name = self.QDRANT_COLLECTION,
            query_vector = question_vector,
            limit = 1
        )
        
        logging.info(f"search result: {search_result}")
        answer = search_result[0].payload['url']

        
        return answer
    
    def import_from_url(self): 
        self.create_collection_if_not_exists()
        self.import_data_if_empty()
    
    def import_data_if_empty(self) -> None:
        """
        Fetchs json data from URL, ads vectors for metadata-infor and imports it to Qdrant if collection is empty
        """
        aidevs_collection = self.qdrant_client.get_collection(collection_name=self.QDRANT_COLLECTION)

        if aidevs_collection and aidevs_collection.points_count > 0:
            logging.info(f"collection {self.QDRANT_COLLECTION} is not empty, skipping import")
            return

        metadata = AssigmentUtils.process_request(self.URL).json()
        metadata = metadata[:1003]
        uuid_list = [str(uuid4()) for _ in range(len(metadata))]
        
        info_list = [single_doc['info'] for single_doc in metadata]
        vector_list = OpenAIEmbeddings().embed_documents(info_list)
        
        points=models.Batch(
            ids = uuid_list, 
            payloads = metadata,
            vectors = vector_list
        )
        logging.info(f"importing {len(points.ids)} points to {self.QDRANT_COLLECTION}")
        try:
            self.qdrant_client.upsert(collection_name=self.QDRANT_COLLECTION, points=points, wait=False)
        except Exception as e:
            logging.error(f"error while importing to qdrant: {e}")
        return None
        
        
    def create_collection_if_not_exists(self) -> None:
    
        """ 
        Creates collection if not exists
        """
        collections = self.qdrant_client.get_collections()
        logging.info(f"searching for collection {self.QDRANT_COLLECTION} in {collections.collections}")
        indexed = False
        for collection in collections.collections:
            if collection.name == self.QDRANT_COLLECTION:
                indexed = True
                break
        if not indexed:
            logging.info(f"creating collection {self.QDRANT_COLLECTION}")
            self.qdrant_client.create_collection(
                collection_name=self.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=1536, distance=models.Distance.COSINE, on_disk=True
                ),
                shard_number=2
            )
        else:
            logging.info(f"collection {self.QDRANT_COLLECTION} already exists")
        
        return None
