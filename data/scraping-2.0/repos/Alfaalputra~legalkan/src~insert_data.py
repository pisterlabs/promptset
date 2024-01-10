from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus

class Insert():

    def __init__(self):
        self.model_name = "firqaaa/indo-sentence-bert-base"
        self.model = HuggingFaceEmbeddings(self.model_name)

    
    def insert_data(self, data):
        vector_db = Milvus.from_documents(
            collection_name="legalKan",
            documents=data,
            embedding=self.model,
            connection_args={"host": "127.0.0.1", "port": "19530"},
        )

        return print("Data Saved to legalKan collection")