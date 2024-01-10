from typing import Dict, List
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.data_structs import Node
    
    
class EmbedNodes:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={"device": "cpu"},
            encode_kwargs={"device": "cpu", "batch_size": 100}
            )
    
    def __call__(self, node_batch: List[Dict[str, Node]]) -> List[Dict[str, Node]]:
        # Debug check
        if not all(isinstance(item, dict) and 'node' in item for item in node_batch):
            raise ValueError("node_batch should be a list of dictionaries with 'node' key")

        nodes = [item['node'] for item in node_batch]
        text = [node.text for node in nodes]
        embeddings = self.embedding_model.embed_documents(text)
        assert len(nodes) == len(embeddings)

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return [{"embedded_nodes": node} for node in nodes]
