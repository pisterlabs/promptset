from langchain.embeddings.base import Embeddings
from typing import List
from zeitghost.vertex.Helpers import rate_limit, _get_api_key, VertexModels
from vertexai.preview.language_models import TextEmbeddingModel


class VertexEmbeddings(Embeddings):
    """
    Helper class for getting document embeddings
    """
    model: TextEmbeddingModel
    project_id: str
    location: str
    requests_per_minute: int
    _api_key: str

    def __init__(self
                 , project_id='cpg-cdp'
                 , location='us-central1'
                 , model=VertexModels.MODEL_EMBEDDING_GECKO.value
                 , requests_per_minute=15):
        """
        :param project_id: str
            Google Cloud Project ID
        :param location: str
            Google Cloud Location
        :param model: str
            LLM Embedding Model name
        :param requests_per_minute: int
            Rate Limiter for managing API limits
        """
        super().__init__()
        
        self.model = TextEmbeddingModel.from_pretrained(model)
        self.project_id = project_id
        self.location = location
        self.requests_per_minute = requests_per_minute
        # self._api_key = _get_api_key()

    def _call_llm_embedding(self, prompt: str) -> List[List[float]]:
        """
        Retrieve embeddings from the embeddings llm

        :param prompt: str
            Document to retrieve embeddings

        :return: List[List[float]]
        """
        embeddings = self.model.get_embeddings([prompt])
        embeddings = [e.values for e in embeddings] #list of list
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Retrieve embeddings for a list of documents

        :param texts: List[str]
            List of documents for embedding

        :return: List[List[float]
        """
        # print(f"Setting requests per minute limit: {self.requests_per_minute}\n")
        limiter = rate_limit(self.requests_per_minute)
        results = []
        for doc in texts:
            chunk = self.embed_query(doc)
            results.append(chunk)
            rate_limit(self.requests_per_minute)
            next(limiter)
        return results
        
    def embed_query(self, text) -> List[float]:
        """
        Retrieve embeddings for a singular document

        :param text: str
            Singleton document

        :return: List[float]
        """
        single_result = self._call_llm_embedding(text)
        # single_result = self.embed_documents([text])
        return single_result[0] #should be a singleton list
