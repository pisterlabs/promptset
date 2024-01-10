from typing import Dict, List
from langchain.embeddings import VertexAIEmbeddings
from pydantic import root_validator


class VertexAIEmbeddingsAIGateway(VertexAIEmbeddings):
    """Google Cloud VertexAI embedding models."""
    api_key: str

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        return values
    
    def embed_documents(
        self, 
        texts: List[str], 
        batch_size: int = 5,
        request_id: str = "",
    ) -> List[List[float]]:
        """Embed a list of strings. Vertex AI currently
        sets a max batch size of 5 strings.

        Args:
            texts: List[str] The list of strings to embed.
            batch_size: [int] The batch size of embeddings to send to the model

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        for batch in range(0, len(texts), batch_size):
            text_batch = texts[batch : batch + batch_size]
            embeddings_batch = self.client.get_embeddings(
                texts=text_batch,
                request_id=request_id,
                api_key=self.api_key,
            )
            embeddings.extend([el.values for el in embeddings_batch])
        return embeddings
    
    def embed_query(
            self, 
            text: str,
            request_id: str = "",
        ) -> List[float]:
        """Embed a text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embeddings = self.client.get_embeddings(
            texts=[text],
            request_id=request_id,
            api_key=self.api_key,
        )
        return embeddings[0].values