import datetime
import os
import shutil

from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

from cli import Color, console


class Cauldron:
    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.persist_directory = './.sapphire'
        self.embedding_function = OpenAIEmbeddings()
        self.__setup_client()

    def reingest(self) -> None:
        self.db = self.__get_new_client()

    def get_db(self):
        return self.db

    def __setup_client(self) -> None:
        if self.__should_reingest():
            self.db = self.__get_new_client()
        else:
            self.db = self.__get_existing_client()

    def __should_reingest(self) -> bool:
        """
        Reingest if cache doesn't exist or last updated != today
        """
        try:
            with open(f'{self.persist_directory}/date.txt', 'r') as f:
                cached_date = f.readline().strip()
                return cached_date != f'{datetime.date.today()}'
        except IOError:
            return True

    def __get_existing_client(self):
        client = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
        )
        return client

    def __get_new_client(self):
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        docs = self.__get_docs()
        client = Chroma.from_documents(
            docs, self.embedding_function, persist_directory=self.persist_directory
        )

        with open(f'{self.persist_directory}/date.txt', 'w') as f:
            f.write(f'{datetime.date.today()}')
        return client

    def __get_docs(self, chunk_size=1000, chunk_overlap=20) -> list:
        console.print(f'{Color.SYSTEM.value}:leaves: Gathering ingredients :leaves:')
        loader = DirectoryLoader(self.directory, glob='**/*.md', show_progress=True)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        console.print(f'{Color.SYSTEM.value}:cocktail: Brewing cauldron :cocktail:')
        docs = text_splitter.split_documents(documents)
        return docs
