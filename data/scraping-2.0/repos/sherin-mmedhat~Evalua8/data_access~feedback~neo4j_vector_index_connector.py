from langchain.vectorstores import Neo4jVector
from langchain.schema import Document
from decouple import config
from ..embedding.embedding_provider import EmbeddingProvider
from model.feedback_metadata import FeedbackMetadata

embedding_provider = EmbeddingProvider()
feedback_node_label = "Feedback"

class Neo4jVectorIndexConnector:
    def __init__(self):
        _uri = config('NEO4J_URI')
        _user = config('NEO4J_USERNAME')
        _password = config('NEO4J_PASSWORD')
        self.vector_store = Neo4jVector(
            embedding_provider.embedding_function(),
            url=_uri,
            username=_user,
            password=_password,
            node_label=feedback_node_label,
        )
    
    def add_feedback(self, feedback: str, evaluator_id: int, kpis: [str]):
        docs = [
            Document(
                page_content=feedback,
                metadata=FeedbackMetadata(evaluator_id = evaluator_id, kpis = kpis),
            ),
        ]
        ids = self.vector_store.add_documents(docs)
        return ids[0]
