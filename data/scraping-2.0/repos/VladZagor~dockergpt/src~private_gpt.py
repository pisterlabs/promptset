import glob
import os
from typing import List

from config import Config
from doc_loader import load_documents
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


class PrivateGPT:
    def __init__(self) -> None:
        self.config = Config()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embeddings_model_name)
        self._init_chroma()

        self.chromaDb = Chroma(persist_directory=self.config.db_directory,
                               embedding_function=self.embeddings,
                               client_settings=self.config.chrome_settings)

        retriever = self.chromaDb.as_retriever(
            search_kwargs={"k": self.config.taget_source_chunks})

        # Change this value based on your model and your GPU VRAM pool.
        n_gpu_layers = 40
        # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_batch = 512
        # Prepare the LLM
        match self.config.model_type:
            case "LlamaCpp":
                # Llama to use GPU
                llm = LlamaCpp(model_path=self.config.model_path,
                               n_ctx=self.config.model_n_ctx,
                               callbacks=[],
                               n_gpu_layers=n_gpu_layers,
                               n_batch=n_batch,
                               verbose=False)
            case "GPT4All":
                # GPT4All to use CPU
                llm = GPT4All(model=self.config.model_path,
                              n_ctx=self.config.model_n_ctx,
                              backend='gptj',
                              callbacks=[],
                              verbose=False)
            case _:
                print(f"Model {self.config.model_type} not supported!")
                exit

        self.retrivalQa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    def ask(self, query):
        """ Query the model and return the answer and the source documents"""
        res = self.retrivalQa(query)
        answer, docs = res['result'], res['source_documents']
        return answer, docs

    def update_documents(self):
        """ Update the vectorstore with new documents in the source directory"""
        docs = self._load_documents(True)
        self.chromaDb.add_documents(docs)
        self.chromaDb.persist()

    def _init_chroma(self):
        if self._does_vectorstore_exist():
            self.chromaDb = Chroma(persist_directory=self.config.db_directory,
                                   embedding_function=self.embeddings,
                                   client_settings=self.config.chrome_settings)
        else:
            docs = self._load_documents(True)
            self.chromaDb = Chroma.from_documents(docs,
                                                  self.embeddings,
                                                  persist_directory=self.config.db_directory,
                                                  client_settings=self.config.chrome_settings)
            self.chromaDb.persist()

    def _load_documents(self, filter_existing=False) -> List[Document]:
        """ Load documents from the source directory and split in chunks """
        ignored_files = [metadata['source'] for metadata in self.chromaDb.get(
        )['metadatas']] if not filter_existing else []
        # Load documents and split in chunks
        documents = load_documents(self.config.source_directory, ignored_files)
        if not documents:
            print("No new documents to load")
            return

        print(
            f"Loaded {len(documents)} new documents from {self.config.source_directory}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        print(
            f"Split into {len(texts)} chunks of text (max. {self.config.chunk_size} tokens each)")
        return texts

    def _does_vectorstore_exist(self) -> bool:
        """
        Checks if vectorstore exists
        """
        if os.path.exists(os.path.join(self.config.db_directory, 'index')):
            if os.path.exists(os.path.join(self.config.db_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(self.config.db_directory, 'chroma-embeddings.parquet')):
                list_index_files = glob.glob(
                    os.path.join(self.config.db_directory, 'index/*.bin'))
                list_index_files += glob.glob(
                    os.path.join(self.config.db_directory, 'index/*.pkl'))
                # At least 3 documents are needed in a working vectorstore
                if len(list_index_files) > 3:
                    return True
        return False
