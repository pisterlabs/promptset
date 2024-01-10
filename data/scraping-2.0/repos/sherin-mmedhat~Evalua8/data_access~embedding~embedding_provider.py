from langchain.embeddings import HuggingFaceBgeEmbeddings

class EmbeddingProvider:
    def __init__(self):
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        query_instruction="Represent this sentence for searching relevant passages: "
        self.embedding = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_instruction=query_instruction,
            cache_folder="models/transformers/"
        )
    
    def embedding_function(self):
        return self.embedding
