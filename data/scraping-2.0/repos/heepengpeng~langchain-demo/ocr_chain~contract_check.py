import os
import time

import langchain
from langchain import PromptTemplate
from langchain.cache import SQLiteCache
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def init_vector_db():
    loader = TextLoader("data/contract_rule_info.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    return docsearch


class ContractCheck:

    def __init__(self, llm):
        self.docsearch = init_vector_db()
        self.llm = llm

    def run(self, query):
        langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
        docs = self.docsearch.similarity_search("租房合同规则有哪些")[0].page_content
        template = """
            请帮我判断合同是否合规

            你需要按照以下规则 {contract_rule}， 如果没有这些字段，则合同不合规。并返回不合规的原因。

            合同的内容是：
            {contract_content}

            """
        with open("./tmp/contract.txt") as f:
            contract_content = f.read()
        prompt = PromptTemplate(
            input_variables=["contract_rule", "contract_content"],
            template=template
        )
        prompt = prompt.format_prompt(contract_rule=docs, contract_content=contract_content)
        start_time = time.time()
        result = self.llm.predict(prompt.to_string())
        end_time = time.time()
        print(f"Consume time {end_time - start_time}")
        return result