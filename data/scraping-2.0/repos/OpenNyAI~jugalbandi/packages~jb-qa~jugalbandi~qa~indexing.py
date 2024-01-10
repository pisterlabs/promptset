from abc import ABC, abstractmethod
import tempfile
import aiofiles
import openai
from jugalbandi.core.errors import InternalServerException, ServiceUnavailableException
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from jugalbandi.document_collection import (
    DocumentCollection,
    DocumentFormat,
)
import json


class Indexer(ABC):
    @abstractmethod
    async def index(self, document_collection: DocumentCollection):
        pass


class GPTIndexer(Indexer):
    async def index(self, document_collection: DocumentCollection):
        try:
            files = [document_collection.local_file_path(file)
                     async for file in document_collection.list_files()]
            documents = SimpleDirectoryReader(input_files=files).load_data()
            index = VectorStoreIndex.from_documents(documents)
            index_content = index.storage_context.to_dict()
            index_str = json.dumps(index_content)
            await document_collection.write_index_file("gpt-index", "index.json",
                                                       bytes(index_str, "utf-8"))
        except openai.error.RateLimitError as e:
            raise ServiceUnavailableException(
                f"OpenAI API request exceeded rate limit: {e}"
            )
        except (openai.error.APIError, openai.error.ServiceUnavailableError):
            raise ServiceUnavailableException(
                "Server is overloaded or unable to answer your request at the moment."
                " Please try again later"
            )
        except Exception as e:
            raise InternalServerException(e.__str__())


class LangchainIndexer(Indexer):
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4 * 1024, chunk_overlap=0, separators=["\n", ".", ""]
        )

    async def index(self, doc_collection: DocumentCollection):
        source_chunks = []
        counter = 0
        async for filename in doc_collection.list_files():
            content = await doc_collection.read_file(filename, DocumentFormat.TEXT)
            public_text_url = await doc_collection.public_url(filename,
                                                              DocumentFormat.TEXT)
            content = content.decode('utf-8')
            content = content.replace("\\n", "\n")
            for chunk in self.splitter.split_text(content):
                new_metadata = {
                    "source": str(counter),
                    "document_name": filename,
                    "txt_file_url": public_text_url,
                }
                source_chunks.append(
                    Document(page_content=chunk, metadata=new_metadata)
                )
                counter += 1
        try:
            search_index = FAISS.from_documents(source_chunks,
                                                OpenAIEmbeddings(client=""))
            await self._save_index_files(search_index, doc_collection)
        except openai.error.RateLimitError as e:
            raise ServiceUnavailableException(
                f"OpenAI API request exceeded rate limit: {e}"
            )
        except (openai.error.APIError, openai.error.ServiceUnavailableError):
            raise ServiceUnavailableException(
                "Server is overloaded or unable to answer your request at the moment."
                " Please try again later"
            )
        except Exception as e:
            raise InternalServerException(e.__str__())

    async def _save_index_files(
        self, search_index: FAISS, doc_collection: DocumentCollection
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            # save in temporary directory
            search_index.save_local(temp_dir)

            async with aiofiles.open(f"{temp_dir}/index.pkl", "rb") as f:
                content = await f.read()
                await doc_collection.write_index_file("langchain", "index.pkl",
                                                      content)

            async with aiofiles.open(f"{temp_dir}/index.faiss", "rb") as f:
                content = await f.read()
                await doc_collection.write_index_file("langchain", "index.faiss",
                                                      content)
