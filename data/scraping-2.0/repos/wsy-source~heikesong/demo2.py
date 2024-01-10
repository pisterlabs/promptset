import asyncio
import os
import datetime

from azure.storage.blob import ContainerClient
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.text_splitter import CharacterTextSplitter
from openai import InvalidRequestError

from llm.llm import llm
from prompt.Extraction import EXTRACTION_PROMPT


async def run_chain(document, semaphore):
    template = PromptTemplate.from_template(EXTRACTION_PROMPT)
    chain = LLMChain(llm=llm, prompt=template, verbose=True)
    async with semaphore:
        try:
            keywords = await chain.arun(input_documents=document)
        except InvalidRequestError:
            return ""
        return keywords


async def extract_keyword(filename):
    conn_string = os.getenv("STORAGE_CONNECTION_STRING",
                            "DefaultEndpointsProtocol=https;AccountName=cggptsc;AccountKey=rWvHP0XV8ji7QnVDDASpbjApgiixQ/RITbzlF62z7CWPkIXWzi6W5ZJIlf0UXU5/Eg5UTwx13XaB+AStuckFbQ==;EndpointSuffix=core.windows.net")
    container_name = os.getenv("CONTAINER_NAME", "gptfiles")
    blob_loader = AzureBlobStorageFileLoader(conn_str=conn_string, container=container_name,
                                             blob_name=filename)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = blob_loader.load_and_split(splitter)
    start_time = datetime.datetime.now()
    semaphore = asyncio.Semaphore(10)
    task_list = [run_chain(document, semaphore) for document in documents]
    results = await asyncio.gather(*task_list)
    print(results)
    end_time = datetime.datetime.now()
    print(end_time - start_time)


if __name__ == "__main__":
    filename = "main.pdf"
    loop = asyncio.get_event_loop()
    loop.run_until_complete(extract_keyword(filename))
