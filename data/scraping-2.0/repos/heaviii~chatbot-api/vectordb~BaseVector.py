from abc import ABC, abstractmethod
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter


from config import Config


class BaseVector(ABC):
    @abstractmethod
    def create_index_text(self):
        return True
    
    def get_docs(self, doc_path):
        print('get_docs_____get_docs_CTS',doc_path)
        return self.get_docs_RCTS(doc_path)
    
    def get_docs_old(self,doc_path):
        print('get_docs_____',doc_path)
        loader = DirectoryLoader(doc_path, glob='**/*')
        # 将数据转成 document 对象，每个文件会作为一个 document
        documents = loader.load()

        # 初始化加载器
        #text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        # 切割加载的 document
        return text_splitter.split_documents(documents)
    
    def get_docs_CTS(self, doc_path):
        print('get_docs_____',doc_path)
        loader = DirectoryLoader(doc_path, glob='**/*')
        # 将数据转成 document 对象，每个文件会作为一个 document
        documents = loader.load()

        text_splitter_CTS = CharacterTextSplitter(
            separator = "####",
            chunk_size = 500, #chunk_size = 1000,
            chunk_overlap = 0
        )
        # 初始化加载器
        # 切割加载的 document   
        return text_splitter_CTS.split_documents(documents)
    
    def get_docs_RCTS(self, doc_path):
        print('get_docs_____',doc_path)
        loader = DirectoryLoader(doc_path, glob='**/*')
        # 将数据转成 document 对象，每个文件会作为一个 document
        documents = loader.load()
        
        text_splitter_RCTS = RecursiveCharacterTextSplitter(
            chunk_size = 300, #chunk_size = 1000,
            chunk_overlap = 10
        )
        split_docs_RCTS = text_splitter_RCTS.split_documents(documents)
        print(f'RecursiveCharacterTextSplitter documents:{len(split_docs_RCTS)}')
        # 切割加载的 document   
        return split_docs_RCTS