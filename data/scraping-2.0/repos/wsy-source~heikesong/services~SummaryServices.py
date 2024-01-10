import os

import tiktoken
from langchain import PromptTemplate, LLMChain
from langchain.chains.summarize import load_summarize_chain
from llm.llm import llm
from langchain.document_loaders.azure_blob_storage_file import AzureBlobStorageFileLoader
from langchain.text_splitter import CharacterTextSplitter

from prompt.summary import SUMMARY_PROMPT


class SummaryServices:
    conn_string = os.getenv("STORAGE_CONNECTION_STRING",
                            "DefaultEndpointsProtocol=https;AccountName=cggptsc;AccountKey=rWvHP0XV8ji7QnVDDASpbjApgiixQ/RITbzlF62z7CWPkIXWzi6W5ZJIlf0UXU5/Eg5UTwx13XaB+AStuckFbQ==;EndpointSuffix=core.windows.net")
    container_name = os.getenv("CONTAINER_NAME", "gptfiles")

    @classmethod
    def summary_article(cls, content: str):
        # chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
        #
        # blob_loader = AzureBlobStorageFileLoader(conn_str=cls.conn_string, container=cls.container_name,
        #                                          blob_name=file_name)
        # splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=0)
        # documents = blob_loader.load_and_split(splitter)
        # summary_result = chain.run(documents)
        encoding = tiktoken.get_encoding("gpt2")
        question_prompt = PromptTemplate.from_template(SUMMARY_PROMPT)
        format_prompt = question_prompt.format_prompt(input_documents=content)
        num_tokens = len(encoding.encode(format_prompt.to_string()))
        print(num_tokens)
        chain = LLMChain(llm=llm, prompt=question_prompt, verbose=True)
        result = chain.run(str(content))
        return result

    # @classmethod
    # async def test2():
    #     lit2 = []
    #     for i in range(15):
    #         await asyncio.sleep(2)
    #         # print(i,"test2")
    #         lit2.append(i)
    #     return lit2
