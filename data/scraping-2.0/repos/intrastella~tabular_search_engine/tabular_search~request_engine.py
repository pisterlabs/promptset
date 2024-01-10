#!/usr/bin/env python3

import ast
import argparse
import csv
import os
import pandas as pd
import pinecone
import re
import requests
import tiktoken
import yaml

from pathlib import Path
from typing import List
from uuid import uuid4

from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.utilities import BingSearchAPIWrapper
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PreviewModel:

    def __init__(self,
                 openai_api_key: List[str],
                 bing_api_key: List[str],
                 name: str,
                 rows: List[str],
                 columns: List[str] = None,
                 destination_file: str = None):

        """
        Creates a metadata table.
        :param api_key: key of openai key
        :param rows: Request that requires a list as a return. Ex.: List all major oil producers in 2020.
        :param columns: List of all demanded information for each row.
                Ex.: [Size of Company, Company income per year] - for each major oil producers in 2020
        """

        self.name = name
        self.rows = ' '.join(rows)
        self.columns = str(columns) if columns else '[]'
        self.destination = destination_file
        self.openai_api_key = openai_api_key[0]

        os.environ['OPENAI_API_KEY'] = openai_api_key[0]
        os.environ["BING_SUBSCRIPTION_KEY"] = bing_api_key[0]
        os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

        self.turbo = OpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

        self.request = ' '.join(rows) + ' Provide additional information about: ' + ', '.join(columns)
        self.prompt_template = None
        self.vec_retrieve = None

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None

        self.load_templates()

    def get_bing_result(self, num_res: int = 15):
        """

        :param num_res: number of allowed results
        :return:
        """
        search = BingSearchAPIWrapper(k=num_res)
        # txt_res = search.run(self.request)
        data_res = search.results(self.request, num_res)

        urls = [data_res[i]['link'] for i in range(len(data_res))]
        checked_urls = self.check_url_exists(urls)
        loader = WebBaseLoader(checked_urls)
        data = loader.load()
        # data[1].page_content = 'oiuhoci'
        # data[1].metadata = {'source': ..., 'title': ..., 'description': ..., 'language': ... }
        return data

    @staticmethod
    def check_url_exists(urls: List[str]):
        checked_urls = []
        for url in urls:
            try:
                if requests.head(url, allow_redirects=True).status_code == 200:
                    checked_urls.append(url)
            except: pass
        return checked_urls

    @staticmethod
    def retrieve():
        from datasets import load_dataset

        data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')
        return data

    def load_templates(self):
        script_path = Path(__file__).parents[1].resolve()

        with open(script_path / 'prompt_template/example_template.txt', 'r') as file:
            self.template = file.read().replace('\n', ' \n ')

        with open(script_path / 'prompt_template/prefix.txt', 'r') as file:
            self.prefix = file.read().replace('\n', ' \n ')

        with open(script_path / 'prompt_template/suffix.txt', 'r') as file:
            self.suffix = file.read().replace('\n', ' \n ')

        with open(script_path / 'prompt_template/examples.yaml', 'r') as file:
            self.examples = yaml.safe_load(file)

        self.examples = [self.examples[k] for k in self.examples.keys()]

    def get_template(self):
        """
        query_item : rows = user request
        query_entries: columns = add infos
        answer: model answer
        :return:
        """

        example = 'Request Item List: {question} Columns Names: {query_entries} Answer: {answer}'
        ex_string = example.format(**self.examples[0])
        partial_s = self.suffix.format(query_entries=self.columns, context='{context}', question='{question}')
        temp = self.prefix + ' \n The following is an example: ' + ex_string + '  \n  ' + partial_s

        self.prompt_template = PromptTemplate(template=temp,
                                              input_variables=['context', 'question'])

    def retrieval(self):
        data = self.get_bing_result()
        self.vec_retrieve = RAG(data=data, openai_api_key=self.openai_api_key)
        self.vec_retrieve.setup()
        self.vec_retrieve.setup_storage()

    def get_answer(self):
        script_path = Path(__file__).parent.resolve()

        self.retrieval()

        qa_with_sources = self.vec_retrieve.GQA_Source(
            self.turbo,
            self.prompt_template
        )

        res = qa_with_sources.run(self.rows)
        out = self.parse_output(res)
        self.save_csv(out)

        df = pd.DataFrame()
        for col in ast.literal_eval(res[12:])[0].keys():
            df[col] = 0
        for elem in ast.literal_eval(res[12:]):
            df.loc[len(df)] = elem

        if self.destination:
            df.to_excel(self.destination, index_label='id')
        else:
            df.to_excel(script_path / f'output_tables/{self.name}.xlsx', index_label='id')

    def save_csv(self, output):
        script_path = Path(__file__).parent.resolve()

        with open(script_path / f'output_csv/{self.name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(output)

    @staticmethod
    def parse_output(output):
        parsed_out = ''
        column_names = None
        while output:
            m = re.search('{(.+?)}', output)
            if m:
                parsed_out += '{' + m.group(1) + '}'
                output = output[m.span()[1]:]

                if not column_names:
                    column_names = list(ast.literal_eval(parsed_out).keys())
            else:
                output = ''

        return '[' + str(column_names) + parsed_out + ']'

    def run(self):
        self.get_template()
        self.get_answer()


class RAG:

    def __init__(self, data, openai_api_key):
        self.data = data
        self.openai_api_key = openai_api_key
        self.tokenizer = tiktoken.get_encoding('p50k_base')
        # os.environ['OPENAI_API_KEY'] = openai_api_key[0]

        self.embed = None
        self.index = None
        self.vectorstore = None

        self.index_name = 'langchain-retrieval-augmentation'

    def setup(self):
        # sub_data = self.data[6]['text']
        sub_data = self.data[0].page_content

        text_splitter = self.divide_txt()
        chunks = text_splitter.split_text(sub_data)[:3]
        embed = self.get_embedding(chunks)
        self.pinecone_index(embed, new=True)

        self.add_data_2_index()

    def tiktoken_len(self, text: str):
        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def divide_txt(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=self.tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter

    def get_embedding(self, texts: List[str]):
        model_name = 'text-embedding-ada-002'

        self.embed = OpenAIEmbeddings(
            document_model_name=model_name,
            query_model_name=model_name,
            openai_api_key=self.openai_api_key
        )

        return self.embed.embed_documents(texts)

    def pinecone_index(self, embedded_txt, new=False):
        pinecone.init(
            api_key="1ea8696f-0f2c-4510-bc3f-1e198071b2b0",
            environment="gcp-starter"
        )

        if new:
            pinecone.delete_index(name=self.index_name)

        pinecone.create_index(
            name=self.index_name,
            metric='dotproduct',
            dimension=len(embedded_txt[0])
        )

        self.get_index()

    def get_index(self):
        # connect to the new index
        self.index = pinecone.GRPCIndex(self.index_name)
        self.index.describe_index_stats()

    def setup_storage(self):
        text_field = "text"
        index = pinecone.Index(self.index_name)

        self.vectorstore = Pinecone(
            index, self.embed.embed_query, text_field
        )

    def add_data_2_index(self):
        batch_limit = 100
        texts = []
        metadatas = []

        # data[1].page_content = 'oiuhoci'
        # data[1].metadata = {'source': ..., 'title': ..., 'description': ..., 'language': ... }

        for i, record in enumerate(self.data):

            if (record.metadata['source'].split('.')[-1] != 'pdf') and ('title' in record.metadata.keys()):
                metadata = {
                    'id': str(i),
                    'source': record.metadata['source'],
                    'title': record.metadata['title']
                }

                text_splitter = self.divide_txt()
                # record_texts = text_splitter.split_text(record['text'])
                record_texts = text_splitter.split_text(record.page_content)

                record_metadatas = [{
                    "chunk": j, "text": text, **metadata
                } for j, text in enumerate(record_texts)]

                texts.extend(record_texts)
                metadatas.extend(record_metadatas)

                if len(texts) >= batch_limit:
                    ids = [str(uuid4()) for _ in range(len(texts))]
                    embeds = self.get_embedding(texts)
                    self.index.upsert(vectors=zip(ids, embeds, metadatas))
                    texts = []
                    metadatas = []

    def similarity_search(self, query):
        self.vectorstore.similarity_search(
            query,
            k=3
        )

    def GQA(self, query, llm, prompt_template):
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs=dict(prompt=prompt_template)
        )
        return qa.run(query)

    def GQA_Source(self, llm, prompt_template):
        chain_type_kwargs = {"prompt": prompt_template}
        qa_with_sources = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=self.vectorstore.as_retriever(),
                                         chain_type_kwargs=chain_type_kwargs)

        '''qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs=dict(prompt=prompt_template)
        )'''

        return qa_with_sources
