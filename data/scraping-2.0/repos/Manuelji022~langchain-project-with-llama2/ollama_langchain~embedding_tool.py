from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama

class ollama_embeddings():
    """Get the embeddings for the text.
        Parameters:
            text (str): The text to get embeddings for.
            model (str): The model to use for embeddings. In this case, llama2:7b.
    """
    def __init__(self):
        self.model = "llama2:7b"

    def get_embeddings(self, text):
        embeddings_model = OllamaEmbeddings(model=self.model)
        embeddings = embeddings_model.embed_query(text)
        return embeddings