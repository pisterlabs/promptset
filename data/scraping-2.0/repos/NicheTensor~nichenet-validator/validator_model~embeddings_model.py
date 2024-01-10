import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy

class EmbeddingsModel:
    def __init__(self, model_to_use, auth_key=None):
        
        self.auth_key = auth_key

        self.set_model_to_use(model_to_use)
        

    def set_model_to_use(self, model_to_use):
        self.model_to_use = model_to_use
        if self.model_to_use == "spacy":
            try:
                self.nlp = spacy.load('en_core_web_md')
            except:
                spacy.cli.download("en_core_web_md")
                self.nlp = spacy.load('en_core_web_md')
            
            self.embedding_function = self.embed_using_spaCy

        elif self.model_to_use == "openai":
            import openai
            self.openai = openai
            openai.api_key = self.auth_key
            self.embedding_function = self.embed_using_openai


    def embed_using_spaCy(self, texts):
        embeddings = [doc.vector for doc in self.nlp.pipe(texts, disable=["tagger", "parser", "ner", "lemmatizer"])]
        return embeddings

    def embed_using_openai(self, texts, model="text-embedding-ada-002"):
        texts = [text.replace("\n", " ") for text in texts]

        embeddings_dict = self.openai.Embedding.create(input = texts, model=model)
        embeddings = [embedding['embedding'] for embedding in embeddings_dict['data']]
        return embeddings


    def embed_texts(self, texts):
        return self.embedding_function(texts)


    def compute_pairwise_similarity(self, texts):
        # Embed each sentence
        embeddings = self.embed_texts(texts)

        # Compute pairwise cosine similarities in a batch
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix