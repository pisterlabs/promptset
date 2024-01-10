import os

import tiktoken
from langchain import PromptTemplate, LLMChain
import requests
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.text_splitter import CharacterTextSplitter

from llm.llm import llm
from tools.DownloadArticleTool import DownloadArticleTool
from tools.LoadDataTool import LoadDataTool
from prompt.LoadDataPrompt import LOAD_DATA_PREFIX, LOAD_DATA_SUFFIX
from prompt.SearchDOI import SEARCH_DOI_PROMPT
from util.TempStore import TempStore


class LoadDataService:
    @classmethod
    def load_data(cls, question):
        try:
            print("=======================================")
            prompt = PromptTemplate.from_template(SEARCH_DOI_PROMPT)
            chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
            answer = chain.run(question)
            print("====================")
            print(answer)
            if answer == "False":
                TempStore.content = question
            else:
                response = requests.get(f"http://20.247.106.150:7052/api/Downloadpdf?doi={answer}")
                response_content_str = response.content.decode('utf-8')
                print(response_content_str)
                key = response_content_str.__contains__("true")
                if key:
                    content = response_content_str.replace(" ","").split("/")
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
                    return "DOI: "+answer+" 我已经下载并分析了这篇文档,您现在可以向我询问这篇文档的信息。"
                else:
                    TempStore.content = question
        except Exception as e:
            print(e.__str__())
            TempStore.content = question
        return "我已经分析完了这篇文档,现在您可以向我询问这篇文档的信息。"
