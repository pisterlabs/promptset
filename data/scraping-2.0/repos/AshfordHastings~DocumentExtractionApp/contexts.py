from typing import Type, List

from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import VectorStore, FAISS

from langchain.document_loaders.pdf import PyPDFLoader


class DocumentArtifact:
    def get_documents():
        raise NotImplementedError("Not Implemented.")

class LocalPDFDocumentArtifact:
    def __init__(self, path, loader_args:dict=None):
        self.path = path
        self.loader_args = loader_args or {}
        self.documents = None
    def load_documents(self):
        self.documents = PyPDFLoader(self.path, **self.loader_args).load()
        return self.documents
    def get_documents(self):
        return self.documents or self.load_documents()

    

# class BusinessDocument:
#     def __init__(self,
#         client_id=None,
#         collection_id=None,
#         object_name=None,
#         artifact=None
#     ):
#         self.object_name = object_name
#         self.artifact = artifact
    


class ExtractionContext:
    def __init__(self):
        pass

    def get_runnable(self):
        raise NotImplementedError("Not Implemented.")
    
    # def invoke(self, *args, **kwargs):
    #     return self.get_runnable().invoke(*args, **kwargs)


class DocumentExtractionContext(ExtractionContext):
    def __init__(self, artifact:DocumentArtifact=None, vector_store:VectorStore=None, embeddings=None):
        self.artifact = artifact
        self.embeddings_class = type(embeddings) if embeddings else OpenAIEmbeddings
        self.vector_store_class = type(vector_store) if vector_store else FAISS 
        self.vector_store = vector_store or None
        self.embeddings = embeddings or None

    def get_runnable(self, artifact=None):
        if artifact: self.artifact = artifact
        if not self.embeddings: self._init_embeddings()
        if not self.vector_store: self._init_vector_store()
        return self.vector_store.as_retriever()
    
    def _init_embeddings(self):
        self.embeddings = self.embeddings_class()

    def _init_vector_store(self):
        self.vector_store =  self.vector_store_class.from_documents(
            documents=self.artifact.get_documents(),
            embedding=self.embeddings,
            #text_splitter=CharacterTextSplitter()
        )




class StringExtractionContext(ExtractionContext):
    def __init__(self, data:list[str]=None, vector_store:VectorStore=None, embeddings=None):
        self.data = data or [""]
        self.embeddings_class = type(embeddings) or OpenAIEmbeddings
        self.vector_store_class = type(vector_store) or FAISS 
        self.vector_store = vector_store or None
        self.embeddings = embeddings or None

    def get_runnable(self, data:list[str]=None):
        if data: self.data = data
        if not self.embeddings: self._init_embeddings() 
        if not self.vector_store: self._init_vector_store()
        return self.vector_store.as_retriever()
    
    def _init_embeddings(self):
        self.embeddings = self.embeddings_class()

    def _init_vector_store(self):
        return self.vector_store_class.from_texts(
            texts=self.data,
            embedding=self.embeddings,
            #text_splitter=CharacterTextSplitter()
        )
