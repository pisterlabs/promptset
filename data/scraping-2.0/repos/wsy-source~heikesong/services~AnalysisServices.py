import logging
import os
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Redis

from llm.llm import llm_embedding


class AnalysisFile:
    conn_string = os.getenv("STORAGE_CONNECTION_STRING",
                            "DefaultEndpointsProtocol=https;AccountName=cggptsc;AccountKey=rWvHP0XV8ji7QnVDDASpbjApgiixQ/RITbzlF62z7CWPkIXWzi6W5ZJIlf0UXU5/Eg5UTwx13XaB+AStuckFbQ==;EndpointSuffix=core.windows.net")
    container_name = os.getenv("CONTAINER_NAME", "gptfiles")
    redis_url = os.getenv("REDIS_URL", "redis://4.236.196.174:6379")

    @classmethod
    def analysis_file(cls, file_name: str, doi, container_name=None):
        logging.info(f"start analysis file {file_name}")
        if container_name == None:
            container_name = cls.container_name
        try:
            blob_loader = AzureBlobStorageFileLoader(conn_str=cls.conn_string, container=container_name,
                                                     blob_name=file_name)
            splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
            documents = blob_loader.load_and_split(splitter)
            logging.info(f"start split file {file_name} completed ,count: {len(documents)}")
            Redis.from_documents(documents=documents, index_name=doi, embedding=llm_embedding,
                                 redis_url=cls.redis_url)
        except Exception as e:
            logging.error("redis embedding error: " + e.__str__())
            return False, e.args.__str__()
        else:
            return True, "success"
