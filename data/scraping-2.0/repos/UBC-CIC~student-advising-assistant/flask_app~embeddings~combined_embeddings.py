from langchain.embeddings.base import Embeddings
from typing import List

class CombinedEmbeddings(Embeddings):
    """
    Embeddings wrapper class that combines precomputed embeddings
    into concatenated embeddings
    """

    query_separator: str = '|'

    def __init__(self, base_model: Embeddings, d: int):
        """
        - base_model: the base embeddings model
        - d: number of embeddings to concatenate
        """
        self.base_model = base_model
        self.d = d

    def concat_embeddings(self, embeddings: List[List[List[float]]]) -> List[List[float]]:
        """
        Create a list of concatenated embeddings from a list of embeddings
        - embeddings: List of precomputed embeddings (d x n x e) 
                        - d is the number of different embeddings
                        - n is the number of documents
                        - e is the embedding dimension
        Outputs a list of embeddings concatenated by document, (n x (d*e))
        """
        assert(len(embeddings) == self.d)
        return [sum(embed_list,[]) for embed_list in zip(*embeddings)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Not implemented and not necessary for this purpose since
        we are loading precomputed embeddings
        Function included for the LangChain Embedding interface
        """
        return []

    def embed_query(self, text: str) -> List[float]:
        """
        Embed query text.
        If text split by the query_separator has dimension d,
        concatenates embeddings for each split portion
        Otherwise, concatenates entire text and concatenates to
        itself d times
        """
        texts = text.split(self.query_separator)
        if len(texts) == self.d:
            query_embeds = [self.base_model.embed_query(text_split) for text_split in texts]
            return sum(query_embeds,[])
        else:
            query_embed = self.base_model.embed_query(text)
            return sum([query_embed for _ in range(self.d)],[])