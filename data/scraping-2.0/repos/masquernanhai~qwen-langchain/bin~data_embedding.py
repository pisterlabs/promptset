from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.chroma import Chroma
import sys,os
class processing_data():
    def __init__(self,text2vec_model_path, load_directory, save_directory):
        self.text2vec_model_path = text2vec_model_path
        self.load_directory = load_directory
        self.save_directory = save_directory

    def load_documents(self):
        loader = DirectoryLoader(self.load_directory)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size = 512, chunk_overlap = 0)
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    def load_embedding_model(self):
        encode_kwargs = {"normalize_embeddings":False}
        model_kwargs = {"device": "cuda:0"}
        return HuggingFaceBgeEmbeddings(
            cache_folder = self.text2vec_model_path,
            model_kwargs = model_kwargs,
            encode_kwargs = encode_kwargs
        )

    def store_chroma(self, docs, embeddings):
        persist_directory = self.save_directory
        db = Chroma.from_documents(docs, embeddings, persist_directory = persist_directory)
        db.persist()
        return db

    def running(self):
        print(self.load_directory)
        print(self.save_directory)
        embeddings = self.load_embedding_model()
        if not os.path.exists(self.save_directory):
            documents = self.load_documents()
            db = self.store_chroma(documents, embeddings)
        else:
            db = Chroma(persist_directory = self.save_directory,embedding_function=embeddings)
        return db



