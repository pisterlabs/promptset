import json
import pickle

from langchain.callbacks import get_openai_callback
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders.pdf import UnstructuredPDFLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from utils import (
    OPENAI_API_KEY,
    DB_PATH,
    DOCS_CACHE_PATH,
    HISTORY_PATH,
    PROMPT_CACHE_PATH,
    make_tempfile,
    compute_file_checksum,
    compute_checksum,
)


QA_PROMPT_TEXT = """
You are an AI assistant for answering questions about a provided set of
documents.
You are given the following extracted parts of a long document and a question.
Provide a conversational answer. If you don't know the answer, just say
"I do not know the answer.". Don't try to make up an answer.

Question: {question}
==========
{context}
==========
Answer in Markdown:
"""
QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEXT, input_variables=['question', 'context'])

CONDENSE_QUESTION_PROMPT_TEXT = """
Given the following conversation and a follow up question, rephrase the
follow up question to be a standalone question.
You can assume the question to be related to the documents.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    CONDENSE_QUESTION_PROMPT_TEXT)

DOC_PROMPT_TEXT = 'Content: {page_content}\nSource: {source}'
DOC_PROMPT = PromptTemplate(
    template=DOC_PROMPT_TEXT,
    input_variables=['page_content', 'source'],
)


class DocumentManager:

    document = None
    filename = None
    checksum = None
    chunks = None

    def get_from_uploaded_file(self, source):
        temp = make_tempfile(source)
        self.checksum = compute_file_checksum(temp)
        self.file_name = source.name.strip()

        docs_cache = DOCS_CACHE_PATH.joinpath(f'{self.checksum}-doc.pkl')
        if docs_cache.exists():
            with docs_cache.open(mode='rb') as fh:
                self.document = pickle.load(fh)
                print(f'Document "{docs_cache}" loaded from disk.')
        else:
            print(f'source.type -> {source.type}')
            if 'pdf' in source.type:
                loader = UnstructuredPDFLoader(temp)
            elif 'markdown' in source.type:
                loader = UnstructuredMarkdownLoader(temp)
            elif 'csv' in source.type:
                loader = CSVLoader(temp)
            elif 'spreadsheetml' in source.type or 'ms-excel' in source.type:
                loader = UnstructuredExcelLoader(temp)
            else:
                loader = TextLoader(temp, encoding='utf-8')

            self.document = loader.load()
            with docs_cache.open(mode='wb') as fh:
                pickle.dump(self.document, fh)
                print(f'Document "{docs_cache}" saved to disk.')

        self._process_chunks()
        temp.unlink()

    def _process_chunks(self):
        chunk_size = 1024
        chunk_overlap = 256

        chunks_cache = DOCS_CACHE_PATH.joinpath(f'{self.checksum}-chunks.pkl')
        if chunks_cache.exists():
            with chunks_cache.open(mode='rb') as fh:
                self.chunks = pickle.load(fh)
                print(f'Chunks "{chunks_cache}" loaded from disk.')
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            self.chunks = text_splitter.split_documents(self.document)
            for i, chunk in enumerate(self.chunks):
                chunk.metadata['source'] = f'{self.file_name}: {i}-pl'
            with chunks_cache.open(mode='wb') as fh:
                pickle.dump(self.chunks, fh)
                print(f'Chunks "{chunks_cache}" saved to disk.')


class VectorStoreManager:

    vectorestore = None
    _documents = None

    def __init__(self, documents: list, name: str = None):
        self._documents = documents
        self.name = compute_checksum(
            name or '-'.join([i.checksum for i in self._documents])
        )
        self.storage = DB_PATH.joinpath(self.name + '.pkl')

        self._process_vectorstore()

    def _process_vectorstore(self):
        if self.storage.exists():
            with self.storage.open('rb') as fh:
                self.vectorestore = pickle.load(fh)
                print(f'Embeddings "{self.name}" loaded from disk.')
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            chunks = []
            for doc in self._documents:
                chunks.extend(doc.chunks)
            self.vectorestore = FAISS.from_documents(
                chunks,
                embedding=embeddings
            )
            with self.storage.open(mode='wb') as fh:
                pickle.dump(self.vectorestore, fh)
                print(f'Embeddings "{self.name}" saved to disk.')


class ChainManager:

    _vectorstore = None
    _history = None
    _qa_prompt = QA_PROMPT
    _doc_prompt = DOC_PROMPT
    _condense_question_prompt = CONDENSE_QUESTION_PROMPT
    _llm_model_name = 'gpt-3.5-turbo-16k'
    _llm_temperature = 0.7
    llm = None
    chain = None
    memory = None

    def __init__(self, vectorstore):
        self._vectorstore = vectorstore

        vs_name = self._vectorstore.name
        self._history = HISTORY_PATH.joinpath(f'{vs_name}.pkl')
        self._init_memory()
        self.init_llm()
        self.init_chain()

    def _init_memory(self):
        if self._history.exists():
            with self._history.open(mode='rb') as fh:
                self.memory = pickle.load(fh)
            print(f'Chat history "{self._history}" loaded from disk.')
        else:
            self.memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True
            )
            self._save_memory()

    def _save_memory(self):
        with self._history.open(mode='wb') as fh:
            pickle.dump(self.memory, fh)
        print(f'Chat history "{self._history}" saved to disk.')

    def get_history(self):
        memory = self.memory.dict()
        chat_memory = memory.get('chat_memory') or {}
        messages = chat_memory.get('messages') or []
        history = []
        for i, msg in enumerate(messages):
            if i % 2 == 0:
                history.append({
                    'role': 'user',
                    'content': msg.get('content') or '',
                })
            else:
                content = json.loads(msg['content'])
                history.append({
                    'role': 'assistant',
                    'content': content.get('answer'),
                    'sources': content.get('sources'),
                })
        return history

    def init_llm(self, model_name: str = None, temperature: float = None):
        if model_name:
            self._llm_model_name = model_name
        if temperature:
            self._llm_temperature = temperature

        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=self._llm_model_name,
            temperature=self._llm_temperature,
        )
        if self.chain:
            self.init_chain()

    def init_chain(
        self,
        qa_prompt: str = None,
        condense_question_prompt: str = None,
        doc_prompt: str = None,
    ):
        if qa_prompt:
            self._qa_prompt = PromptTemplate(
                template=qa_prompt, input_variables=['question', 'context'])
        if doc_prompt:
            self._doc_prompt = PromptTemplate(
                template=doc_prompt,
                input_variables=['page_content', 'source'],
            )
        if condense_question_prompt:
            self._condense_question_prompt = PromptTemplate.from_template(
                condense_question_prompt
            )

        vectorestore = self._vectorstore.vectorestore

        qa_chain = create_qa_with_sources_chain(
            llm=self.llm,
            prompt=self._qa_prompt,
        )
        final_qa_chain = StuffDocumentsChain(
            llm_chain=qa_chain,
            document_variable_name='context',
            document_prompt=self._doc_prompt,
        )
        condense_question_chain = LLMChain(
            llm=self.llm,
            prompt=self._condense_question_prompt,
        )

        self.chain = ConversationalRetrievalChain(
            question_generator=condense_question_chain,
            retriever=vectorestore.as_retriever(),
            memory=self.memory,
            combine_docs_chain=final_qa_chain,
        )

    def query(self, question):
        with get_openai_callback() as cb:
            response = self.chain({'question': question})
            self._save_memory()
            print(f'get_response callback - {cb}')
        return response


class PromptManager:
    qa_prompts = None
    cq_prompts = None
    _selected_qa_prompt_idx = None
    _selected_cq_prompt_idx = None

    _vectorstore = None
    _qa_cache = None
    _qas_cache = None
    _cq_cache = None
    _cqs_cache = None

    def __init__(self, vectorstore):
        self._vectorstore = vectorstore
        vs_name = self._vectorstore.name
        self._qa_cache = PROMPT_CACHE_PATH.joinpath(f'{vs_name}-qa.pkl')
        self._qas_cache = PROMPT_CACHE_PATH.joinpath(f'{vs_name}-qas.pkl')
        self._cq_cache = PROMPT_CACHE_PATH.joinpath(f'{vs_name}-cq.pkl')
        self._cqs_cache = PROMPT_CACHE_PATH.joinpath(f'{vs_name}-cqs.pkl')
        self._init_cache()

    def _init_cache(self):
        if self._qa_cache.exists() and self._qas_cache.exists():
            with self._qa_cache.open(mode='rb') as fh:
                self.qa_prompts = pickle.load(fh)
                print(f'QA Prompts "{self._qa_cache}" loaded from disk.')
            with self._qas_cache.open(mode='rb') as fh:
                self._selected_qa_prompt_idx = pickle.load(fh)
                print(f'Selected QA Prompt "{self._qas_cache}" '
                      f'loaded from disk.')
        else:
            self.qa_prompts = []
            self.add_qa_prompt(QA_PROMPT_TEXT)
            self.set_qa_prompt(0)
            with self._qa_cache.open(mode='wb') as fh:
                pickle.dump(self.qa_prompts, fh)
                print(f'QA Prompts "{self._qa_cache}" saved to disk.')
            with self._qas_cache.open(mode='wb') as fh:
                pickle.dump(self._selected_qa_prompt_idx, fh)
                print(f'Selected QA Prompt "{self._qas_cache}" '
                      f'saved to disk.')

        if self._cq_cache.exists() and self._cqs_cache.exists():
            with self._cq_cache.open(mode='rb') as fh:
                self.cq_prompts = pickle.load(fh)
                print(f'Condensed Question Prompts "{self._cq_cache}" '
                      f'loaded from disk.')
            with self._cqs_cache.open(mode='rb') as fh:
                self._selected_cq_prompt_idx = pickle.load(fh)
                print(f'Selected Condensed Question Prompt '
                      f'"{self._cqs_cache}" loaded from disk.')
        else:
            self.cq_prompts = []
            self.add_cq_prompt(CONDENSE_QUESTION_PROMPT_TEXT)
            self.set_cq_prompt(0)
            with self._cq_cache.open(mode='wb') as fh:
                pickle.dump(self.cq_prompts, fh)
                print(f'Condensed Question Prompts "{self._cq_cache}" '
                      f'saved to disk.')
            with self._cqs_cache.open(mode='wb') as fh:
                pickle.dump(self._selected_cq_prompt_idx, fh)
                print(f'Selected Condensed Question Prompt '
                      f'"{self._cqs_cache}" saved to disk.')

    def set_qa_prompt(self, prompt_idx: int):
        self._selected_qa_prompt_idx = prompt_idx
        with self._qas_cache.open(mode='wb') as fh:
            pickle.dump(self._selected_qa_prompt_idx, fh)
            print(f'Selected QA Prompt "{self._qas_cache}" '
                  f'saved to disk.')

    def add_qa_prompt(self, prompt: str):
        has_question = '{question}' in prompt
        has_context = '{context}' in prompt

        if has_question and has_context:
            self.qa_prompts.append(prompt)
            with self._qa_cache.open(mode='wb') as fh:
                pickle.dump(self.qa_prompts, fh)
                print(f'QA Prompts "{self._qa_cache}" saved to disk.')
        else:
            raise ValueError('Missing question or context placeholders.')

    @property
    def qa_prompt(self):
        return self.qa_prompts[self._selected_qa_prompt_idx]

    @property
    def qa_prompt_index(self):
        return self._selected_qa_prompt_idx

    def set_cq_prompt(self, prompt_idx: int):
        self._selected_cq_prompt_idx = prompt_idx
        with self._cqs_cache.open(mode='wb') as fh:
            pickle.dump(self._selected_cq_prompt_idx, fh)
            print(f'Selected Condensed Question Prompt '
                  f'"{self._cqs_cache}" saved to disk.')

    def add_cq_prompt(self, prompt: str):
        has_question = '{question}' in prompt
        has_history = '{chat_history}' in prompt

        if has_question and has_history:
            self.cq_prompts.append(prompt)
            with self._cq_cache.open(mode='wb') as fh:
                pickle.dump(self.cq_prompts, fh)
                print(f'Condensed Question Prompts "{self._cq_cache}" '
                      f'saved to disk.')
        else:
            raise ValueError('Missing question or chat history placeholders.')

    @property
    def cq_prompt(self):
        return self.cq_prompts[self._selected_cq_prompt_idx]

    @property
    def cq_prompt_index(self):
        return self._selected_cq_prompt_idx
