import langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, Docx2txtLoader, UnstructuredPDFLoader, SeleniumURLLoader
from bilibili import BiliBiliLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import VectorStore, FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_gemini_chat_models import ChatGoogleGenerativeAI

import os
import hashlib
import json
import traceback
import asyncio
from logging import Handler
from enum import Enum
from typing import List, Tuple, Optional
from fastapi import UploadFile


class ModelType(Enum):
    GPT = 1,
    GEMINI = 2


class LangchainClient:
    gpt35_token = 6000
    gpt4_token = 3000
    gemini_token = 12000

    langchain.verbose = True
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=['\n\n', '\n', ' ', ''], model_name='gpt-3.5-turbo-16k', chunk_size=gpt35_token / 2, chunk_overlap=150)
    faiss_dir = 'faissSave/'
    file_dir = 'files/'

    use_gpt4 = True

    file_context_prefix = 'f:'
    url_context_prefix = 'u:'
    bilibili_context_prefix = 'b:'
    text_context_prefix = 't:'
    context_prefix = [file_context_prefix, url_context_prefix,
                      bilibili_context_prefix, text_context_prefix]

    summarize_prompt_prefix = ':s'
    special_prompt_prefix = [summarize_prompt_prefix]

    def __init__(self, openai_api_key: str, google_api_key: str, embeddingLogger: Handler, gptLogger: Handler, geminiLogger: Handler):
        self.embeddingLogger = embeddingLogger
        self.gptLogger = gptLogger
        self.geminiLogger = geminiLogger
        self.update_openai_api_key(openai_api_key)
        self.update_google_api_key(google_api_key)

    def update_openai_api_key(self, key: str):
        os.environ['OPENAI_API_KEY'] = key
        self.embeddings = OpenAIEmbeddings(client=None)
        self.llm35 = ChatOpenAI(model='gpt-3.5-turbo-16k',
                                temperature=0.7, max_tokens=self.gpt35_token)
        self.llm4 = ChatOpenAI(
            model='gpt-4', temperature=0.7, max_tokens=self.gpt4_token)

    def update_google_api_key(self, key: str):
        os.environ['GOOGLE_API_KEY'] = key
        self.llm_gemini = ChatGoogleGenerativeAI(
            model='gemini-pro', temperature=0.7, max_output_tokens=self.gemini_token, convert_system_message_to_human=True)

    async def request(self, messages: List, type: ModelType) -> Tuple[str, str]:
        if messages[0].get('role') == 'system' and messages[0].get('content').startswith(tuple(self.context_prefix)):
            if messages[0].get('content').startswith(self.file_context_prefix):
                result_content, source_content = await self.file_base_request(messages, type)
            elif messages[0].get('content').startswith(self.bilibili_context_prefix):
                result_content, source_content = await self.bilibili_base_request(
                    messages, type)
            elif messages[0].get('content').startswith(self.text_context_prefix):
                result_content, source_content = await self.text_base_request(messages, type)
            else:
                result_content, source_content = await self.url_base_request(messages, type)
        else:
            result_content, source_content = await self.langchain_request(messages, type)

        return result_content, source_content

    async def langchain_request(self, messages: List, type: ModelType) -> Tuple[str, str]:
        contents = []
        messages.reverse()
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'user':
                message = HumanMessage(content=content)
            elif role == 'assistant':
                message = AIMessage(content=content)
            else:
                if type == ModelType.GPT:
                    message = SystemMessage(content=content)
                elif type == ModelType.GEMINI:
                    continue
            contents.append(message)

            if type == ModelType.GPT:
                if self.use_gpt4 and self.llm4.get_num_tokens_from_messages(contents) > self.gpt4_token:
                    break
                if not self.use_gpt4 and self.llm35.get_num_tokens_from_messages(contents) > self.gpt35_token:
                    break
            elif type == ModelType.GEMINI:
                if self.llm_gemini.get_num_tokens_from_messages(contents) > self.gemini_token:
                    break

        if type == ModelType.GEMINI:
            for content in contents[::-1]:
                if not isinstance(content, HumanMessage):
                    del contents[-1]
                else:
                    break

        contents.reverse()

        if type == ModelType.GPT:
            if self.use_gpt4:
                result = await self.llm4.agenerate([contents])
            else:
                result = await self.llm35.agenerate([contents])
        elif type == ModelType.GEMINI:
            result = await self.llm_gemini.agenerate([contents])

        return result.generations[0][0].text, ''

    async def based_request(self, messages: List, db: VectorStore, index: str, type: ModelType) -> Tuple[str, str]:
        query = messages[-1].get('content')
        if query.startswith(tuple(self.special_prompt_prefix)):
            if query.startswith(self.summarize_prompt_prefix):
                return await self.summarize_based_request(index, query, type)
            else:
                return await self.summarize_based_request(index, query, type)
        else:
            return await self.conversational_based_request(messages, db, type)

    async def conversational_based_request(self, messages: List, db: VectorStore, type: ModelType) -> Tuple[str, str]:
        if type == ModelType.GPT:
            llm = self.llm35
            limit = self.gpt35_token*1.2
        elif type == ModelType.GEMINI:
            llm = self.llm_gemini
            limit = self.gemini_token*1.2

        _template = """通过给出的对话历史和追加的问题, 改写追加的问题成为一个独立的问题, 用对话历史的语言。
        对话历史:
```
        {chat_history}
```
        追加的问题: {question}
        独立的问题:"""
        condense_question_prompt = PromptTemplate.from_template(_template)

        system_template = """根据下文内容回答问题。如果无法回答，回复“我不知道”，不要编造答案。
```
        {context}
```
        """
        combine_docs_chain_messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        combine_docs_chain_prompt = ChatPromptTemplate.from_messages(
            combine_docs_chain_messages)

        qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(search_type='mmr'), chain_type='stuff',
                                                   return_source_documents=True, return_generated_question=True,
                                                   max_tokens_limit=limit, condense_question_prompt=condense_question_prompt,
                                                   combine_docs_chain_kwargs={'prompt': combine_docs_chain_prompt})

        chat_history = []
        i = 1
        while i < len(messages) - 1:
            msg = messages[i]
            role = msg.get('role')
            query = msg.get('content')
            i += 1
            if role == 'user':
                msg = messages[i]
                role = msg.get('role')
                if role == 'assistant':
                    answer = msg.get('content')
                    chat_history.append((query, answer))
                    i += 1

        query = messages[-1].get('content')
        content = {'question': query, 'chat_history': chat_history}
        result = await qa.acall(content)
        result_content = result['answer']
        source_content = ''
        try:
            generated_question = result["generated_question"]
            source_content = generated_question

            source_docs = result['source_documents']
            contexts = []
            for doc in source_docs:
                contexts.append(doc.page_content)
            source_content += '\n\n' + '\n\n'.join(contexts)
        except Exception as e:
            traceback.print_exc()
            if type == ModelType.GPT:
                self.gptLogger.exception(e)
            elif type == ModelType.GEMINI:
                self.geminiLogger.exception(e)
        return result_content, source_content

    async def summarize_based_request(self, index: str, query: str, type: ModelType) -> Tuple[str, str]:
        if type == ModelType.GPT:
            llm = self.llm35
        elif type == ModelType.GEMINI:
            llm = self.llm_gemini

        loader = TextLoader(f'{self.faiss_dir}{index}/{index}.txt',
                            autodetect_encoding=True)
        data = loader.load()
        docs = self.text_splitter.split_documents(data)
        prompt = query[len(self.summarize_prompt_prefix):]

        map_template = """详细总结下文各段落的内容，如果无法总结则重复下文全部内容，忽略无法总结的部分:
```
        {text}
```
        你的回答:"""
        if len(prompt.strip()) > 0:
            map_template = '通过下文内容，' + prompt + '，如果无法回答则重复下文全部内容，忽略无法总结的部分' + """:
```
            {text}
```
            你的回答:"""
        map_prompt = PromptTemplate(
            template=map_template, input_variables=["text"])

        combine_template = """精要地重复下文全部内容，忽略无法总结的部分:
```
        {text}
```
        你的回答:"""
        if len(prompt.strip()) > 0:
            combine_template = '通过下文内容，详细说明' + prompt + '，如果无法说明则重复下文全部内容，忽略无法总结的部分' + """:
```
            {text}
 ```
            你的回答:"""
        combine_prompt = PromptTemplate(
            template=combine_template, input_variables=["text"])

        chain = load_summarize_chain(llm, chain_type="map_reduce",
                                     map_prompt=map_prompt, combine_prompt=combine_prompt, token_max=self.gpt35_token)
        result = await chain.arun(docs)

        source_content = ''
        try:
            contexts = []
            for doc in docs:
                contexts.append(doc.page_content)
            source_content = '\n\n'.join(contexts)
        except Exception as e:
            traceback.print_exc()
            if type == ModelType.GPT:
                self.gptLogger.exception(e)
            elif type == ModelType.GEMINI:
                self.geminiLogger.exception(e)
        return result, source_content

    async def file_base_request(self, messages: List, type: ModelType) -> Tuple[str, str]:
        content = messages[0].get('content')
        context = content[len(self.file_context_prefix):]
        db = FAISS.load_local(self.faiss_dir + context, self.embeddings)
        return await self.based_request(messages, db, context, type)

    async def url_base_request(self, messages: List, type: ModelType) -> Tuple[str, str]:
        content = messages[0].get('content')
        url = content[len(self.url_context_prefix):]
        hl = hashlib.md5()
        hl.update(url.encode(encoding='utf-8'))
        context = hl.hexdigest()
        path = self.faiss_dir + context
        if not os.path.exists(path):
            db = await self.load_url(url, context)
        else:
            db = FAISS.load_local(path, self.embeddings)
        return await self.based_request(messages, db, context, type)

    async def bilibili_base_request(self, messages: List, type: ModelType) -> Tuple[str, str]:
        content = messages[0].get('content')
        url = content[len(self.bilibili_context_prefix):]
        hl = hashlib.md5()
        hl.update(url.encode(encoding='utf-8'))
        context = hl.hexdigest()
        path = self.faiss_dir + context
        if not os.path.exists(path):
            db = await self.load_bilibli(url, context)
            if not db:
                return '该视频未生成字幕', ''
        else:
            db = FAISS.load_local(path, self.embeddings)
        return await self.based_request(messages, db, context, type)

    async def text_base_request(self, messages: List, type: ModelType) -> Tuple[str, str]:
        content = messages[0].get('content')
        text = content[len(self.text_context_prefix):]
        hl = hashlib.md5()
        hl.update(text.encode(encoding='utf-8'))
        context = hl.hexdigest()
        path = self.faiss_dir + context
        if not os.path.exists(path):
            data = [Document(page_content=text, metadata={})]
            first_line = text[:text.index('\n')] if '\n' in text else text
            db = await self.save_docs_to_db(data, context, first_line)
        else:
            db = FAISS.load_local(path, self.embeddings)
        return await self.based_request(messages, db, context, type)

    async def load_url(self, url: str, index: str) -> VectorStore:
        loader = SeleniumURLLoader(urls=[url], headless=False)
        data = loader.load()
        db = await self.save_docs_to_db(data, index, url)
        return db

    async def load_bilibli(self, url: str, index: str) -> Optional[VectorStore]:
        cookies = json.loads(
            open('./bili_cookies_0.json', encoding='utf-8').read())
        loader = BiliBiliLoader(video_urls=[url], cookies=cookies)
        data = loader.load()
        text = data[0].page_content
        if (text == ''):
            return None
        db = await self.save_docs_to_db(data, index, url)
        return db

    async def save_docs_to_db(self, data: List[Document], index: str, source: str) -> VectorStore:
        docs = self.text_splitter.split_documents(data)
        loop = asyncio.get_event_loop()
        db = await loop.run_in_executor(None, FAISS.from_documents, docs, self.embeddings)
        db.save_local(self.faiss_dir + index)
        self.embeddingLogger.info(f'{index} - {source}')
        with open(f'{self.faiss_dir}{index}/{index}.txt', 'w', encoding='utf8') as txt:
            for doc in data:
                txt.write(doc.page_content)
                txt.write('\n\n')
            txt.close()
        return db

    async def upload_file(self, file: UploadFile):
        if index == None or len(index.strip()) <= 0:
            hl = hashlib.md5()
            while True:
                content = await file.read(8192)
                if not content:
                    await file.seek(0)
                    break
                hl.update(content)
            index = hl.hexdigest()

        ext = file.filename.split('.')[-1]
        name = self.file_dir + index + '.' + ext
        with open(name, 'wb') as f:
            content = await file.read()
            f.write(content)

        if ext == 'txt':
            loader = TextLoader(name, autodetect_encoding=True)
        elif ext == 'docx' or ext == 'dox':
            loader = Docx2txtLoader(name)
        elif ext == 'pdf':
            loader = UnstructuredPDFLoader(name)
        else:
            return {'message': f'{file.filename} not support', 'index': ''}

        data = loader.load()
        await self.save_docs_to_db(data, index, file.filename)
