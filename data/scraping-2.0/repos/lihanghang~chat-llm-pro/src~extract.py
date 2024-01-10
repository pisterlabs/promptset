"""
基于Langchain构建LLM应用
https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html
https://eyurtsev.github.io/kor/   extract structured data from text using large language models (LLMs) 🧩.
"""
import asyncio
import json
import logging

import requests
from kor import create_extraction_chain, extract_from_documents
from langchain import PromptTemplate, LLMChain, Modal
from langchain.llms import openai
from langchain.schema import Document
from typing import List

from data import prompt_text
from src.utils.doc import parser_doc, read_file, hashcode_with_file
import os
import dotenv

from langchain.chat_models import AzureChatOpenAI

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='[%(pastime)s] {%(pathname)s:%(lineno)d} %(levelness)s - %(message)s',
                    datefmt='%H:%M:%S')
logger.setLevel(logging.INFO)

config = dotenv.dotenv_values(".env")
openai.api_type = config["API_TYPE"]
openai.api_base = config["OPENAI_API_BASE"]
openai.api_version = config["OPENAI_API_VERSION"]

# fix 尽量用这种方式设置azure的可以，测试了下openai_api_key不起作用。
os.environ['OPENAI_API_KEY'] = config["AZURE_OPENAI_API_KEY"]
endpoint_url = os.getenv('MEM_FIN_OPENAI_API')


azure_llm = AzureChatOpenAI(
        deployment_name="gpt-35-turbo",
        temperature=0,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        top_p=1.0
)


class MyModal(Modal):

    def _call(self, prompt: str, stop=None) -> str:
        body = {"prompt": prompt, "max_length": 2048, "temperature": 0.2}
        response = requests.post(self.endpoint_url, json=body)
        return response.json()['response']


def parser_pdf(file_path) -> List[Document]:
    logging.info("加载并解析文件……")
    file_name = file_path.name
    hashcode = hashcode_with_file(file_name)
    parser_file = parser_doc(file_name, f'data/store/{hashcode}')
    contents: list = read_file(parser_file)
    txt = ''.join(contents)
    docs = [Document(page_content=txt)]
    return docs


def extract_doc(schema, llm_type, docs=[]):
    """
    基于大模型进行文档信息抽取
    """
    mem_llm = MyModal(endpoint_url=endpoint_url)
    llm_obj = mem_llm if llm_type == 'memect' else azure_llm            # azure_llm
    logging.info("Create extract chain")
    extraction_chain = create_extraction_chain(llm_obj, schema, encoder_or_encoder_class='json')
    logging.info("Start extract from doc……")
    extraction_results = asyncio.run(
        extract_from_documents(
            chain=extraction_chain,
            documents=docs,
            use_uid=False,
            max_concurrency=2
        )
    )
    ret = json.dumps(extraction_results[0]['data'], ensure_ascii=False, indent=4)
    logging.info(ret)
    return ret


def chat_mem_fin_llm(endpoint_url, input_text, task_type):
    """
    基于langchain调用memect LLM openapi
    """
    mem_llm = MyModal(endpoint_url=endpoint_url)
    query = f"{prompt_text[task_type]} {input_text.strip()}"
    prompt_template = PromptTemplate(input_variables=["query"],
                                     template=f'{{query}}')
    llm_chain = LLMChain(llm=mem_llm, prompt=prompt_template)
    response = llm_chain.run(query)
    return response


if __name__ == '__main__':
    pass
