from langchain.vectorstores import DeepLake
from config.wiki_writer_config import AppConfig

class DeepLakeProvider():
    def __init__(self, db):
        self.db = db

    @staticmethod
    def instance(dataset_path, embeddings, read_only=False):
        db = DeepLake(
            token=AppConfig.ACTIVE_LOOP_TOKEN,
            dataset_path=dataset_path,
            embedding_function=embeddings,
            read_only=read_only
        )
        return DeepLakeProvider(
            db=db
        )
    
    def add_documents(self, docs):
        print('DOCS', docs)
        self.db.add_documents(docs)
        return
    
    def as_retriever(self):
        return self.db.as_retriever()