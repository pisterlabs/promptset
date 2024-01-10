import ray 
from starlette.requests import Request
from ray import serve
from typing import List
# from embeddings import LocalHuggingFaceEmbeddings 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import time

FAISS_INDEX_PATH = 'faiss_index' 

@serve.deployment
class VectorSearchDeployment:
    def __init__(self):
        #Load the data from faiss
        st = time.time()
        # self.embeddings = LocalHuggingFaceEmbeddings('multi-qa-mpnet-base-dot-v1')
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        self.db = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings)
        et = time.time() - st
        print(f'Loading database took {et} seconds.')

    def search(self, query):
        results = self.db.max_marginal_relevance_search(query)
        retval = ''
        for i in range(len(results)):
            chunk = results[i]
            retval += chunk.page_content
            retval = retval + '\n====\n\n'
                           
        return retval
    
    async def __call__(self, request: Request) -> List[str]:
        return self.search(request.query_params["query"])

deployment = VectorSearchDeployment.bind()