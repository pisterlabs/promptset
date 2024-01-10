import os
from typing import Any

import tiktoken
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool
import requests

from util.TempStore import TempStore


class DownloadArticleTool(BaseTool):
    name = "DownloadArticleTool"
    description = """useful tool to download article 
                    If user input contains doi you should use this tool
                    If the user enters the subject, abstract doi etc. you should use this tool
                    parameter: doi  You need to extract based on user input such as  "**.****/8888-020-0636-z"
                """

    def _run(self, doi: str) -> Any:
        response = requests.get(f"http://20.247.106.150:7052/api/Downloadpdf?doi={doi}")
        response_content_str = response.content.decode('utf-8')
        print(response_content_str)
        key = response_content_str.__contains__("true")
        if key:
            content = response_content_str.split("/")
            temp_name = content[len(content) - 1]
            file_name = temp_name.split("?")[0].replace("@", "-")
            conn_string = os.getenv("STORAGE_CONNECTION_STRING",
                                    "DefaultEndpointsProtocol=https;AccountName=cggptsc;AccountKey=rWvHP0XV8ji7QnVDDASpbjApgiixQ/RITbzlF62z7CWPkIXWzi6W5ZJIlf0UXU5/Eg5UTwx13XaB+AStuckFbQ==;EndpointSuffix=core.windows.net")
            container_name = os.getenv("CONTAINER_NAME", "gptfiles")
            blob_loader = AzureBlobStorageFileLoader(conn_str=conn_string, container=container_name,
                                                     blob_name=file_name)
            splitter = CharacterTextSplitter()
            document = blob_loader.load_and_split(splitter)
            print("下载完成")
            encoding = tiktoken.get_encoding("gpt2")
            num_tokens = len(encoding.encode(str(document)))
            while num_tokens > 27000:
                document.pop()
                num_tokens = len(encoding.encode(str(document)))
                print(num_tokens)
            TempStore.content = str(document)


            message = f"{doi} 下载成功,您现在可以询问有关这篇文章的内容"
        else:
            message = f"{doi} 下载失败"
        return message

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        pass
