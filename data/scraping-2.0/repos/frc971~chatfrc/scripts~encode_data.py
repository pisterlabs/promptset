"""
    This script reads documents from data/ and encodes them into a data.npy file
"""

import os
import threading
from langchain.document_loaders.base import BaseLoader

import numpy as np
from dotenv import load_dotenv

from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredPDFLoader, UnstructuredRSTLoader, UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings

from langchain.text_splitter import TokenTextSplitter


class CustomDataLoader:

    def __init__(self, chunk_size=500, parent_chunk_size=1000, child_chunk_size=100, do_parent_document=False, thread=False) -> None:
        '''
        if do_parent_document = true, than the dataloader will generate a parent documents and child documents from the data
        parent_chunk_size and child_chunk_size are parameters of do_parent_document
        '''
        self.files = []
        self.thread = thread
        self.documents = []

        self.do_parent_document = do_parent_document

        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-ada-002")

        self.text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        
        self.parent_splitter = TokenTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=0)

        self.child_splitter = TokenTextSplitter(chunk_size=child_chunk_size, chunk_overlap=0)

        self.counter = 0

        self.lock = threading.Lock()

        self.semaphore = threading.Semaphore(3)
    
    def embed_documents(self) -> None:
        self._load_files()

        for file in self.files:
            name, _ = os.path.splitext(file) 

            if name == "data/links":
                continue
            
            if self.thread:
                threading.Thread(target=self._process_document, args=(file,)).start()
                self.semaphore.acquire()
            else:
                self._process_document(file)

    def save(self) -> None:
        print(len(self.documents))
        np.save('data.npy', np.asarray(self.documents, dtype=object))
    
    
    def _generate_child_docs(self, parent_document):
        child_docs = self.child_splitter.split_documents([parent_document])
        vector_responses = self.embeddings_model.embed_documents(
            list(map(lambda document: document.page_content, child_docs))
        )
        for doc in child_docs:
            doc.page_content = parent_document.page_content
        document_map = []
        for doc in zip(vector_responses, child_docs):
            document_map.append({
                "vector": doc[0],
                "document": doc[1]
            })
        return document_map

    
    def _process_document(self, file) -> None:
        _, extension = os.path.splitext(file)

        loader = TextLoader(file)

        match extension:
            case ".pdf":
                loader = UnstructuredPDFLoader(file)
            case ".rst":
                loader = UnstructuredRSTLoader(file)
            case ".md":
                loader = UnstructuredMarkdownLoader(file)


        documents = self.text_splitter.split_documents(loader.load())
        parent_documents = self.parent_splitter.split_documents(loader.load())
        vector_responses = self.embeddings_model.embed_documents(
            list(map(lambda document: document.page_content, documents))
        )

        document_map = []

        for doc in zip(vector_responses, documents):
            document_map.append({
                "vector": doc[0],
                "document": doc[1]
            })
        if self.do_parent_document:
            for doc in parent_documents:
                document_map.extend(self._generate_child_docs(doc))

        with self.lock:
            print('\r', f'Embedding progress: {self.counter + 1}/{len(self.files) - 1}')
            self.documents.extend(document_map)

            self.counter += 1
        
        self.semaphore.release()


    def _load_files(self) -> None:
        for root, _, files in os.walk('data/'):
            for filename in files:
                full_path = os.path.join(root, filename)
                name, extension = os.path.splitext(full_path) 

                match extension:
                    case ".pdf" | ".rst" | ".md":
                        pass
                    case _:
                        # Links is a special case, its where we load arbitrary html
                        if (name != "data/links"):
                            continue

                self.files.append(full_path)

def main():
    load_dotenv()
    """
        Loading multiple file types like this isn't ideal, in the future 
        we'd want to make our own DirectoryLoader class that can handle
        multiple filetypes better and can check embeddings against pinecone.
    """

    loader = CustomDataLoader()

    loader.embed_documents()

    loader.save()

if __name__ == "__main__":
    main()
