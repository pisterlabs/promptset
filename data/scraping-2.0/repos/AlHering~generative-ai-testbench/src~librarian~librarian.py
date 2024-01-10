# -*- coding: utf-8 -*-
"""
****************************************************
*           generative_ai_testbench:librarian                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import functools
from typing import List, Tuple, Any
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from multiprocessing import Pool
from tqdm import tqdm
from src.utility import langchain_utility


def reload_document(document_path: str) -> Document:
    """
    Function for (re)loading document content.
    :param document_path: Document path.
    :return: Document object.
    """
    res = langchain_utility.DOCUMENT_LOADERS[os.path.splitext(document_path)[
        1]](document_path).load()
    return res[0] if isinstance(res, list) and len(res) == 1 else res


class Librarian(object):
    """
    Class, representing an LLM-based librarian agent to interactively query document libraries.
    """

    def __init__(self, profile: dict) -> None:
        """
        Initiation method.
        :param profile: Profile, configuring a librarian agent. The profile should be a nested dictionary of the form
            'llm': General LLM used for interaction.
            'chromadb_settings': ChromaDB Settings.
            'embedding_function': Embedding function.
            'retrieval_source_chunks': Source chunks for retrieval.
            'splitting_chunks': Chunk size for text splitting. Optional.
            'splitting_overlap': Overlap for text splitting. Optional.
        """
        self.profile = profile
        self.vector_db = langchain_utility.get_or_create_chromadb(
            profile["chromadb_settings"], profile["embedding_function"])
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": profile["retrieval_source_chunks"]})
        self.llm = profile["llm"]

    def reload_folder(self, folder: str) -> None:
        """
        Method for (re)loading folder contents.
        :param folder: Folder path.
        """
        document_paths = []
        documents = []
        for root, dirs, files in os.walk(folder, topdown=True):
            document_paths.extend([os.path.join(root, file) for file in files if any(file.lower().endswith(
                supported_extension) for supported_extension in langchain_utility.DOCUMENT_LOADERS)])

        with Pool(processes=os.cpu_count()) as pool:
            with tqdm(total=len(document_paths), desc="(Re)loading folder contents...", ncols=80) as progress_bar:
                for index, loaded_document in enumerate(pool.imap(reload_document, document_paths)):
                    documents.append(loaded_document)
                    progress_bar.update(index)

        if "splitting_chunks" in self.profile:
            documents = self.split_documents(documents)
        self.add_documents_to_db(documents)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Method for splitting document content.
        :param documents: Documents to split.
        :return: Split documents.
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self.profile["splitting_chunks"],
            chunk_overlap=self.profile["splitting_overlap"],
            length_function=len).split_documents(documents)

    def add_documents_to_db(self, documents: List[Document]) -> None:
        """
        Method for adding document contents to DB.
        :param documents: Documents.
        """
        langchain_utility.add_documents_to_chromadb(
            self.vector_db, documents)

    def query(self, query: str, include_source: bool = True, override_llm: Any = None, override_reciever: Any = None) -> Tuple[str, List[Document]]:
        """
        Method for querying for answer.
        :param query: Query.
        :param include_source: Flag, declaring whether to show source documents.
        :param override_llm: Optional LLM to override standard.
        :param override_retriever: Optional Retriever to override standard.
        :return: Answer and list of source documents as tuple.
        """
        qa = RetrievalQA.from_chain_type(
            llm=self.llm if override_llm is None else override_llm,
            retriever=self.retriever if override_reciever is None else override_reciever,
            return_source_documents=include_source)

        response = qa(query)
        return response["result"], response["source_documents"] if include_source else []
