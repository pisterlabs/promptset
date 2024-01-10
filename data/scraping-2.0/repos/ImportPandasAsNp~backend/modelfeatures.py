from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import string

MODEL = None
EMBDIM = 576

class ModelEmbeddings():
    def __init__(self):
        pass

    @staticmethod
    def getModel():
        global MODEL
        if MODEL is None:
            MODEL = SentenceTransformerEmbeddings(model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1")
        return MODEL
    
    @staticmethod
    def preprocessText(text):
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()

        return text
    
    @staticmethod
    def getEmbeddings(text):
        global EMBDIM
        model = ModelEmbeddings.getModel()
        text = ModelEmbeddings.preprocessText(text)
        text = model.embed_query(text)

        text.extend([ 0.0 for _ in range(EMBDIM-512)])
 
        return text