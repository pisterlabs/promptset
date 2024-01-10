import shutil
import sys
import random
from functools import partial
import glob
from filelock import FileLock, Timeout
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
import re
from semanticscholar import SemanticScholar
from semanticscholar.SemanticScholar import Paper
from langchain.utilities import BingSearchAPIWrapper
from collections import Counter
import mmh3
from pprint import pprint
import time
import concurrent.futures
import pandas as pd
import tiktoken
from copy import deepcopy, copy
from collections import defaultdict
import requests
import tempfile
from tqdm import tqdm
import requests
import dill
import os
import re
from prompts import prompts
from langchain.document_loaders import MathpixPDFLoader
from datetime import datetime, timedelta

from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain import OpenAI, ConversationChain
from langchain.embeddings import OpenAIEmbeddings
from review_criterias import review_params
from pathlib import Path
from more_itertools import peekable
from concurrent.futures import Future

import openai
import tiktoken

from web_scraping import fetch_html

try:
    import ujson as json
except ImportError:
    import json


from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index import (
    GPTVectorStoreIndex, 
    LangchainEmbedding, 
    LLMPredictor, 
    ServiceContext, 
    StorageContext, 
    download_loader,
    PromptHelper
)
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext

from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document as LangchainDocument
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from llama_index.data_structs.node import Node, DocumentRelationship
from llama_index import LangchainEmbedding, ServiceContext
from llama_index import GPTTreeIndex, SimpleDirectoryReader
from langchain.document_loaders import PyPDFLoader


from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from typing import Optional, Type
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import BingSearchAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate

from common import *
from base import *
import ai21
from langchain.schema import Document

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('max_colwidth', 800)
pd.set_option('display.max_columns', 100)

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.ERROR,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.getcwd(), "log.txt"))
    ]
)
logger.setLevel(logging.ERROR)
time_logger = logging.getLogger(__name__ + " | TIMING")
time_logger.setLevel(logging.INFO)  # Set log level for this logger

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import asyncio
import threading
from playwright.async_api import async_playwright
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import time

class DocFAISS(FAISS):
    
    def merge_from(self, target: FAISS) -> None:
        """Merge another FAISS object with the current one.

        Add the target FAISS to the current one.

        Args:
            target: FAISS object you wish to merge into the current one

        Returns:
            None.
        """
        from langchain.docstore.base import AddableMixin
        from langchain.schema import Document
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError("Cannot merge with this type of docstore")
        # Numerical index for target docs are incremental on existing ones
        starting_len = len(self.index_to_docstore_id)

        # Merge two IndexFlatL2
        self.index.merge_from(target.index)

        # Get id and docs from target FAISS object
        full_info = []
        existing_id = set([target_id for i, target_id in self.index_to_docstore_id.items()])
        for i, target_id in target.index_to_docstore_id.items():
            if target_id in existing_id:
                continue
            doc = target.docstore.search(target_id)
            if not isinstance(doc, Document):
                raise ValueError("Document should be returned")
            full_info.append((starting_len + i, target_id, doc))

        # Add information to docstore and index_to_docstore_id.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)

def create_index_faiss(chunks, embed_model, doc_id=None):
    from langchain.schema import Document
    if doc_id is None:
        doc_id = [""] * len(chunks)
    elif isinstance(doc_id, (str, int)):
        doc_id = [doc_id] * len(chunks)
    else:
        assert len(doc_id) == len(chunks) and isinstance(doc_id, (list, tuple))
        doc_id = [int(d) for d in doc_id]
    chunks = [Document(page_content=str(c), metadata={"order": i}) for i, c in enumerate(chunks)]
    for ix, chunk in enumerate(chunks):
        chunk.metadata["next"] = None if ix == len(chunks)-1 else chunks[ix + 1]
        chunk.metadata["previous"] = None if ix == 0 else chunks[ix - 1]
        chunk.metadata["doc_id"] = doc_id[ix]
        chunk.metadata["index"] = ix
    db = DocFAISS.from_documents(chunks, embed_model)
    return db


class DocIndex:
    def __init__(self, doc_source, doc_filetype, doc_type, doc_text, full_summary, openai_embed, storage):

        self._visible = False
        self.result_cutoff = 2
        self.version = 0
        self.last_access_time = time.time()
        self.doc_id = str(mmh3.hash(doc_source + doc_filetype + doc_type, signed=False))
        self.doc_source = doc_source
        self.doc_filetype = doc_filetype
        self.doc_type = doc_type
        self._title = ''
        self._short_summary = ''
        folder = os.path.join(storage, f"{self.doc_id}")
        os.makedirs(folder, exist_ok=True)
        self._storage = folder
        self.store_separate = ["indices", "raw_data", "qna_data", "deep_reader_data", "review_data", "static_data", "_paper_details"]
        assert  doc_filetype == "pdf" and ("http" in doc_source or os.path.exists(doc_source))
        self.is_local = os.path.exists(doc_source)
        
        
        static_data = dict(doc_source=doc_source, doc_filetype=doc_filetype, doc_type=doc_type, doc_text=doc_text,)
        raw_data = dict(chunks=full_summary["chunks"], small_chunks=full_summary["small_chunks"])
        raw_index_future = get_async_future(create_index_faiss, raw_data['chunks'], openai_embed, doc_id=self.doc_id,)
        small_chunk_index_future = get_async_future(create_index_faiss, raw_data["small_chunks"], openai_embed,)
        del full_summary["chunks"]
        del full_summary["small_chunks"]
        
        qna_data = dict(chunked_summary=full_summary["chunked_summary"], running_summary=full_summary["running_summary"], detailed_qna=full_summary["detailed_qna"], extended_abstract=dict())
        deep_reader_data = full_summary["deep_reader_details"]
        review_data = []
        _paper_details = None
        # self.set_doc_data("static_data", None, static_data)
        # self.set_doc_data("raw_data", None, raw_data)
        # self.set_doc_data("qna_data", None, qna_data)
        # self.set_doc_data("deep_reader_data", None, deep_reader_data)
        # self.set_doc_data("review_data", None, review_data)
        # self.set_doc_data("_paper_details", None, _paper_details)
        # self.set_doc_data("indices", None, indices)

        futures = [get_async_future(self.set_doc_data, "static_data", None, static_data), get_async_future(self.set_doc_data, "raw_data", None, raw_data), get_async_future(self.set_doc_data, "qna_data", None, qna_data), get_async_future(self.set_doc_data, "deep_reader_data", None, deep_reader_data), get_async_future(self.set_doc_data, "review_data", None, review_data), get_async_future(self.set_doc_data, "_paper_details", None, _paper_details)]
        indices = dict(dqna_index=create_index_faiss([''], openai_embed, doc_id=self.doc_id, ),
                       raw_index=raw_index_future.result(),
                       summary_index=create_index_faiss([''], openai_embed, ),
                       small_chunk_index=small_chunk_index_future.result())
        futures.append(get_async_future(self.set_doc_data, "indices", None, indices))
        for f in futures:
            f.result()
        
        
    @property
    def visible(self):
        return self._visible if hasattr(self, "_visible") else True

    def get_doc_data(self, top_key, inner_key=None,):
        import dill
        doc_id = self.doc_id

        folder = self._storage
        filepath = os.path.join(folder, f"{doc_id}-{top_key}.partial")
        json_filepath = os.path.join(folder, f"{doc_id}-{top_key}.json")
        
        try:
            assert top_key in self.store_separate
        except Exception as e:
            raise ValueError(f"Invalid top_key {top_key} provided")
        logger.info(f"Get doc data for top_key = {top_key}, inner_key = {inner_key}, folder = {folder}, filepath = {filepath} exists = {os.path.exists(filepath)}, json filepath = {json_filepath} exists = {os.path.exists(json_filepath)}, already loaded = {getattr(self, top_key, None) is not None}")
        if getattr(self, top_key, None) is not None:
            if inner_key is not None:
                return getattr(self, top_key, None).get(inner_key, None)
            else:
                return getattr(self, top_key, None)
        else:
            if os.path.exists(json_filepath):
                with open(json_filepath, "r") as f:
                    obj = json.load(f)
                setattr(self, top_key, obj)
                if inner_key is not None:
                    return obj.get(inner_key, None)
                else:
                    return obj
            elif os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    obj = dill.load(f)
                if top_key not in ["indices", "_paper_details"]:
                    with open(json_filepath, "w") as f:
                        json.dump(obj, f)
                setattr(self, top_key, obj)
                if inner_key is not None:
                    return obj.get(inner_key, None)
                else:
                    return obj
            else:
                return None
        
    
    def set_doc_data(self, top_key, inner_key, value, overwrite=False):
        import dill
        doc_id = self.doc_id
        folder = self._storage
        print(folder)
        filepath = os.path.join(folder, f"{doc_id}-{top_key}.partial")
        json_filepath = os.path.join(folder, f"{doc_id}-{top_key}.json")
        path = Path(folder)
        lock_location = os.path.join(os.path.join(path.parent.parent, "locks"), f"{doc_id}-{top_key}")
        lock = FileLock(f"{lock_location}.lock")
        with lock.acquire(timeout=600):
            if top_key == "deep_reader_data":
                if os.path.exists(json_filepath):
                    with open(json_filepath, "r") as f:
                        old_deep_reader_details = json.load(f)
                elif os.path.exists(filepath):
                    with open(os.path.join(filepath), "rb") as f:
                        old_deep_reader_details = dill.load(f)
                else:
                    old_deep_reader_details = dict()

                for k, v in old_deep_reader_details.items():
                    if inner_key is None or k.strip() == inner_key.strip():
                        continue
                    if v is not None and isinstance(v["text"], str)  and len(v["text"].strip()) > 0 and checkNoneOrEmpty(self.get_doc_data("deep_reader_data").get(k, dict()).get("text", None)):
                        self.set_doc_data("deep_reader_data", k, v)
            
            if top_key == "qna_data" and inner_key == "detailed_qna":
                if os.path.exists(json_filepath):
                    with open(json_filepath, "r") as f:
                        old_qna_details = json.load(f)
                elif os.path.exists(filepath):
                    with open(os.path.join(filepath), "rb") as f:
                        old_qna_details = dill.load(f)
                else:
                    old_qna_details = dict()
                    
                current_qid = [d[0] for d in self.get_doc_data("qna_data","detailed_qna") + value]
                if overwrite:
                    current_qna = value
                    for _, (qid, q, a, m) in enumerate(old_qna_details.get("detailed_qna", [])):
                        if len(q.strip()) > 0 and qid not in current_qid:
                            current_qna.append([qid, q, a, m])
                    value = current_qna
                else:
                    current_qna = self.get_doc_data("qna_data","detailed_qna") + value
                    for _, (qid, q, a, m) in enumerate(value):
                        if len(q.strip()) > 0 and qid not in current_qid:
                            value.append([qid, q, a, m])
            
            if inner_key is not None:
                tk = self.get_doc_data(top_key)
                if tk is None:
                    setattr(self, top_key, dict())
                
                inner = self.get_doc_data(top_key, inner_key)
                assert type(inner) == type(value) or inner is None or (isinstance(inner, (tuple, list)) and isinstance(value, (tuple, list)))
                if isinstance(inner, dict) and not overwrite:
                    inner.update(value)
                elif isinstance(inner, list) and not overwrite:
                    inner.extend(value)
                elif isinstance(inner, str) and not overwrite:
                    inner = inner + value
                elif isinstance(inner, tuple) and not overwrite:
                    inner = inner + value
                else:
                    inner = value
                getattr(self, top_key, None)[inner_key] = inner
            else:
                tk = self.get_doc_data(top_key, None)
                if top_key == "review_data" and isinstance(tk, dict):
                    tk = list(tk.values())
                assert (type(tk) == type(value) or tk is None or value is None) or (isinstance(tk, (tuple, list)) and isinstance(value, (tuple, list)))
                if tk is not None and type(tk) == type(value):
                    if isinstance(tk, dict) and not overwrite:
                        tk.update(value)
                    elif isinstance(tk, list) and not overwrite:
                        tk.extend(value)
                    elif isinstance(tk, str) and not overwrite:
                        tk = tk + value
                    elif isinstance(tk, tuple) and not overwrite:
                        tk = tk + value
                    else:
                        tk = value
                    setattr(self, top_key, tk)
                elif tk is None and value is not None:
                    setattr(self, top_key, value)
                else:
                    setattr(self, top_key, None)
            if top_key not in ["indices", "_paper_details"]:
                with open(json_filepath, "w") as f:
                    json.dump(getattr(self, top_key, None), f)
            else:
                with open(os.path.join(filepath), "wb") as f:
                    dill.dump(getattr(self, top_key, None), f)
            
    
    def get_short_answer(self, query, mode=defaultdict(lambda:False), save_answer=True):
        answer = ''
        for ans in self.streaming_get_short_answer(query, mode, save_answer):
            answer += ans
        return answer
    
    @property
    def streaming_followup(self):
        return prompts.streaming_followup

    @property
    def short_streaming_answer_prompt(self):
        return prompts.short_streaming_answer_prompt
    
    @property
    def running_summary_prompt(self):
        return prompts.running_summary_prompt
    
    
    
    def get_date(self):
        paper_details = self.paper_details
        if "publicationDate" in paper_details:
            return paper_details["publicationDate"][:7]
        elif "year" in paper_details:
            return paper_details["year"] + "-01"
        if "arxiv.org" in self.doc_source:
            yr = self.doc_source.split("/")[-1].split(".")[0]
            if is_int(yr):
                return yr
            return None
        return None

    def semantic_search_document(self, query):
        # tldr = (self.paper_details["tldr"] + "\n\n") if "tldr" in self.paper_details and self.paper_details[
        #     "tldr"] is not None and len(self.paper_details["tldr"].strip()) > 0 else ""
        # title = (self.paper_details["title"] + "\n\n") if "title" in self.paper_details and self.paper_details[
        #     "title"] is not None and len(self.paper_details["title"].strip()) > 0 else ""
        # brief_summary = title + tldr + self.short_summary
        # brief_summary = (brief_summary + "\n\n") if len(brief_summary.strip()) > 0 else ""
        brief_summary = ""
        summary_nodes = self.get_doc_data("indices", "summary_index").similarity_search(query, k=self.result_cutoff * 2)
        summary_text = "\n".join([n.page_content for n in summary_nodes])  # + "\n" + additional_text_qna
        summary_text = (summary_text + "\n\n") if len(summary_text.strip()) > 0 else ""
        rem_init_len = 512 * 4
        rem_word_len = rem_init_len - get_gpt3_word_count(
            summary_text + brief_summary)
        rem_tokens = rem_word_len // LARGE_CHUNK_LEN
        raw_nodes = self.get_doc_data("indices", "raw_index").similarity_search(query,
                                                                                k=max(self.result_cutoff, rem_tokens))
        raw_text = "\n".join([n.page_content for n in raw_nodes])
        return brief_summary + summary_text + "\n\n" + raw_text

    
    @streaming_timer
    def streaming_get_short_answer(self, query, mode=defaultdict(lambda:False), save_answer=True):
        ent_time = time.time()
        detail_level = 1
        if mode["provide_detailed_answers"]:
            detail_level = int(mode["provide_detailed_answers"])
            mode = "detailed"
            query = f"{query}\n\nWrite detailed, informative, comprehensive and in depth answer. Provide as much detail, information and depth as possible.\n\n"
        elif mode["review"]:
            mode = "detailed"
            detail_level = 1
        else:
            mode = None
            detail_level = 1

        # Sequential + RAG approach -> then combine.
        # For level 1, 2 both approaches use gpt3.5-16k -> gpt4-16k
        # For level 3, 4 both approaches use gpt3.5-16k + gpt4-16k

        brief_summary = self.title + "\n" + self.short_summary
        brief_summary = ("Summary:\n"+ brief_summary +"\n\n") if len(brief_summary.strip()) > 0 else ""
        additional_info = None
        if mode == "detailed" or mode == "review":
            text = brief_summary + self.get_doc_data("static_data", "doc_text")
            tex_len = get_gpt4_word_count(text)
            if tex_len < 7000:
                llm = CallLLm(self.get_api_keys(), use_gpt4=True, use_16k=False)
                prompt = f"""Answer the question or query given below using the given context as reference. 
Question or Query is given below.
{query}

Context is given below.
{text}

Write answer below.
"""
                additional_info = get_async_future(llm, prompt, temperature=0.5)
            else:
                additional_info = get_async_future(call_contextual_reader, query,
                                                   brief_summary + self.get_doc_data("static_data", "doc_text"),
                                                   self.get_api_keys(), provide_short_responses=False, chunk_size=TOKEN_LIMIT_FOR_DETAILED + 500, scan=detail_level >= 2)

            if detail_level >= 2 and tex_len >= 7000:
                raw_nodes = self.get_doc_data("indices", "raw_index").similarity_search(query, k=max(self.result_cutoff,
                                                                                                     4200//LARGE_CHUNK_LEN))
                raw_text = "\n\n".join([n.page_content for n in raw_nodes])
                small_chunk_nodes = self.get_doc_data("indices", "small_chunk_index").similarity_search(query, k=max(
                    self.result_cutoff, 1200//SMALL_CHUNK_LEN))
                small_chunk_text = "\n\n".join([n.page_content for n in small_chunk_nodes])
                raw_text = raw_text + " \n\n " + small_chunk_text
                prompt = self.short_streaming_answer_prompt.format(query=query, fragment=brief_summary + raw_text, full_summary='')

                llm = CallLLm(self.get_api_keys(), use_gpt4=False, use_16k=True)
                additional_info_v1 = additional_info

                def get_additional_info():
                    ad_info = get_async_future(llm, prompt, temperature=0.8)
                    init_add_info = additional_info_v1.result()
                    return init_add_info + "\n\n" + ad_info.result()
                additional_info = get_async_future(get_additional_info)

        answer = ''
        if detail_level < 2 or additional_info is None or mode=="review":
            llm = CallLLm(self.get_api_keys(), use_gpt4=mode == "detailed" and detail_level > 1, use_16k=True)
            rem_word_len = MODEL_TOKENS_SMART - get_gpt4_word_count(brief_summary) - 2000
            rem_tokens = rem_word_len // LARGE_CHUNK_LEN
            raw_nodes = self.get_doc_data("indices", "raw_index").similarity_search(query, k=max(self.result_cutoff, rem_tokens))
            raw_text = "\n\n".join([n.page_content for n in raw_nodes])
            st_wt = time.time()
            while (additional_info is not None and time.time() - st_wt < 45 and not additional_info.done()):
                time.sleep(0.5)
            full_summary = ""
            if additional_info.done():
                full_summary = additional_info.result() if additional_info is not None else ""
            full_summary = f"Short summary of the document is given below. \n'''{full_summary}'''" if len(full_summary.strip()) > 0 else ""
            rem_word_len = MODEL_TOKENS_SMART - get_gpt4_word_count(brief_summary + raw_text) - 500
            if rem_word_len > SMALL_CHUNK_LEN:
                rem_tokens = rem_word_len // SMALL_CHUNK_LEN
                small_chunk_nodes = self.get_doc_data("indices", "small_chunk_index").similarity_search(query, k=max(self.result_cutoff, rem_tokens))
                small_chunk_text = "\n\n".join([n.page_content for n in small_chunk_nodes])
                raw_text = raw_text + " \n\n " + small_chunk_text
            prompt = self.short_streaming_answer_prompt.format(query=query, fragment=brief_summary+raw_text, full_summary=full_summary)

            if llm.use_gpt4 and llm.use_16k:
                prompt = get_first_last_parts(prompt, 1000, MODEL_TOKENS_SMART*2 - 1000)
            elif llm.use_gpt4:
                prompt = get_first_last_parts(prompt, 1000, MODEL_TOKENS_SMART - 500)
            elif llm.use_16k:
                prompt = get_first_last_parts(prompt, 1000, MODEL_TOKENS_SMART * 2 - 1000)
            else:
                prompt = get_first_last_parts(prompt, 1000, MODEL_TOKENS_DUMB - 1000)

            ans_begin_time = time.time()
            logger.info(f"streaming_get_short_answer:: Start to answer by {(ans_begin_time-ent_time):4f}s")

            main_ans_gen = llm(prompt, temperature=0.7, stream=True)
            for txt in main_ans_gen:
                yield txt
                answer += txt

            yield "</br> \n"
        if mode == "detailed" and detail_level >= 2 and additional_info is not None:
            additional_info = additional_info.result()
            for t in additional_info:
                yield t
                answer += t

        if save_answer:
            get_async_future(self.put_answer, query, answer, mode=mode)
        
    def get_fixed_details(self, key):
        if self.get_doc_data("deep_reader_data") is not None and self.get_doc_data("deep_reader_data", key) is not None and len(self.get_doc_data("deep_reader_data", key)["text"].strip())>0:
            logger.debug(f'Found fixed details for key = {key}')
            return self.get_doc_data("deep_reader_data", key)
        keys = [
                        "methodology",
                        "previous_literature_and_differentiation",
                        "experiments_and_evaluation",
                        "results_and_comparison",
                        "limitations_and_future_work"
                    ]
        assert key in keys
        key_to_query_map = prompts.paper_details_map
        full_text = ''
        for txt in self.streaming_get_short_answer(key_to_query_map[key], defaultdict(lambda: False, {"provide_detailed_answers": True}), save_answer=False):
            full_text += txt
            yield txt
        self.set_doc_data("deep_reader_data", key, {"id": str(mmh3.hash(self.doc_source + key, signed=False)), "text": full_text})
        
    
    def get_short_info(self):
        return dict(visible=self.visible, doc_id=self.doc_id, source=self.doc_source, title=self.title, short_summary=self.short_summary, summary=self.get_doc_data("qna_data", "running_summary") if self.get_doc_data("qna_data", "running_summary") is not None else '')
    
    @property
    def title(self):
        if hasattr(self, "_title") and len(self._title.strip()) > 0:
            return self._title
        else:
            try:
                title = self.paper_details["title"]
            except Exception as e:
                title = CallLLm(self.get_api_keys(), use_gpt4=False)(f"""Provide a title for the below text: \n'{self.get_doc_data("raw_data", "chunks")[0]}' \nTitle: \n""")
            setattr(self, "_title", title)
            self.save_local()
            return title
    
    @staticmethod
    def process_one_paper(paper, extended_abstract):
        string_keys = ["paperId", "venue", "url", "title", "abstract", "tldr", "year", "referenceCount", "citationCount", "journal"]
        keys = ["publicationDate", "citations", "references", "externalIds", ]
        paper_output = dict()
        for k in string_keys:
            paper_output[k] = str(getattr(paper, k))
        
#         print(paper.title, getattr(paper, "publicationDate"), paper.year)
        pubdate = getattr(paper, "publicationDate")
        if pubdate is None:
            paper_output["publicationDate"] = str(paper.year)+"-01-01"
        else:
            paper_output["publicationDate"] = pubdate.strftime("%Y-%m-%d")
        paper_output['ArXiv'] = NoneToDefault(getattr(paper, "externalIds", dict()), dict()).get('ArXiv')
        paper_output["citations"] = [DocIndex.process_one_paper(c, None) for c in NoneToDefault(getattr(paper, "citations", []))]
        paper_output["references"] = [DocIndex.process_one_paper(c, None) for c in NoneToDefault(getattr(paper, "references", []))]
        paper_output["citations"] = [c for c in paper_output["citations"] if c["paperId"] is not None and len(c["paperId"])>0 and c["paperId"].lower()!="none"]
        paper_output["references"] = [c for c in paper_output["references"] if c["paperId"] is not None and len(c["paperId"])>0 and c["paperId"].lower()!="none"]
        paper_output["extended_abstract"] = extended_abstract
        return paper_output
        
    @property
    def paper_details(self)->dict:
        try:
            if hasattr(self, "is_local") and self.is_local or "arxiv.org" not in self.doc_source:
                return dict()
            elif self.get_doc_data("_paper_details") is not None:
                pd = deepcopy(self.get_doc_data("_paper_details"))
                if self.get_doc_data("qna_data", "extended_abstract") is None:
                    self.set_doc_data("qna_data", "extended_abstract", dict())
                extended_abstract = self.get_doc_data("qna_data", "extended_abstract").get(pd["paperId"], None)
                return DocIndex.process_one_paper(pd, extended_abstract)
            else:
                arxiv_url = self.doc_source
                paper = get_paper_details_from_semantic_scholar(arxiv_url)
                self.set_doc_data("_paper_details", None, paper)
                return self.paper_details
        except Exception as e:
            logger.error(f"Error in fetching paper details for {self.doc_source}")
            return dict()
    
    def refetch_paper_details(self)->dict:
        if hasattr(self, "is_local") and self.is_local or "arxiv.org" not in self.doc_source:
            return dict()
        url = self.doc_source
        paper = get_paper_details_from_semantic_scholar(url)
        self.set_doc_data("_paper_details", None, paper)
        return self.paper_details
    
    def get_extended_abstract_for_ref_or_cite(self, paperId)->str:
        if self.get_doc_data("qna_data", "extended_abstract") is None:
            self.set_doc_data("qna_data", "extended_abstract", dict())
        paper_details = self.paper_details
        for ref in paper_details["references"] + paper_details["citations"]:
            if ref["paperId"] == paperId:
                text = self.get_doc_data("qna_data", "extended_abstract").get(paperId, None)
                yield text
                if text.strip() != '':
                    return None
        
        from semanticscholar import SemanticScholar
        sch = SemanticScholar()
        paper = sch.get_paper(paperId)
        if 'ArXiv' in paper.externalIds:
            arxiv = paper.externalIds['ArXiv']
            pdf_url = f"https://arxiv.org/pdf/{arxiv}.pdf"
            data = PDFReaderTool()(pdf_url, page_ranges="1-3")
            prompt = f"""Provide a detailed and comprehensive summary for the scientific text given. This scientific text is the beginning two pages of a larger research paper, as such some details maybe incomplete in this scientific text.
Abstract:
'{paper.abstract}'

Scientific Text:
'{data}'

Detailed and comprehensive summary:

            """
            answer = ''
            for txt in CallLLm(self.get_api_keys(), use_gpt4=False)(prompt, temperature=0.7, stream=True):
                yield txt
                answer += txt
            self.get_doc_data("qna_data", "extended_abstract")[paperId] = answer
            self.set_doc_data("qna_data", "extended_abstract", self.get_doc_data("qna_data", "extended_abstract"))
            
        else:
            yield "Could not find ArXiv pdf for this document"
            
        
    @property
    def short_summary(self):
        if hasattr(self, "_short_summary") and len(self._short_summary.strip()) > 0:
            return self._short_summary
        else:
            try:
                short_summary = self.paper_details["abstract"]
            except Exception as e:
                short_summary = CallLLm(self.get_api_keys(), use_gpt4=False)(f"""Provide a summary for the below scientific text: \n'''{ChunkText(self.get_doc_data("static_data", "doc_text"), TOKEN_LIMIT_FOR_SHORT, 0)[0]}''' \nInclude relevant keywords, the provided abstract and any search/seo friendly terms in your summary. \nSummary: \n""",)
            setattr(self, "_short_summary", short_summary)
            self.save_local()
            return short_summary
        
    
    def get_all_details(self):
        details = dict(chunked_summary=self.get_doc_data("qna_data", "chunked_summary"), 
                       deep_reader_details=self.get_doc_data("deep_reader_data"), 
                       detailed_qna=self.get_doc_data("qna_data", "detailed_qna"), 
                       running_summary=self.get_doc_data("qna_data", "running_summary"))
        
        return dict(doc_id=self.doc_id, source=self.doc_source, title=self.title, short_summary=self.short_summary, summary=self.get_doc_data("qna_data", "running_summary"), details=details)
    
    
    def streaming_ask_follow_up(self, query, previous_answer, mode=defaultdict(lambda: False)):
    
        if mode["provide_detailed_answers"]:
            mode = "detailed"
        else:
            mode = None
        llm = CallLLm(self.get_api_keys(), use_gpt4=True)
        answer = previous_answer["answer"] + "\n" + (
            previous_answer["parent"]["answer"] if "parent" in previous_answer else "")
        rem_word_len = MODEL_TOKENS_SMART - get_gpt4_word_count(answer) - 2000
        rem_tokens = rem_word_len // LARGE_CHUNK_LEN
        raw_nodes = self.get_doc_data("indices", "raw_index").similarity_search(query, k=max(self.result_cutoff, rem_tokens))
        raw_text = "\n".join([n.page_content for n in raw_nodes])
        rem_word_len = MODEL_TOKENS_SMART - get_gpt4_word_count(answer + raw_text) - 500
        rem_tokens = rem_word_len // SMALL_CHUNK_LEN
        small_chunk_nodes = self.get_doc_data("indices", "small_chunk_index").similarity_search(query, k=max(self.result_cutoff * 2, rem_tokens))

        # Get those nodes that don't come up in last query.
        small_chunk_nodes_ids = [n.metadata["order"] for n in small_chunk_nodes]
        small_chunk_nodes_old = self.get_doc_data("indices", "small_chunk_index").similarity_search(previous_answer["query"], k=self.result_cutoff*8)
        small_chunk_nodes_ids = small_chunk_nodes_ids + [n.metadata["order"] for n in small_chunk_nodes_old]
        
        additional_small_chunk_nodes = self.get_doc_data("indices", "small_chunk_index").similarity_search(query, k=self.result_cutoff*8)
        additional_small_chunk_nodes = [n for n in additional_small_chunk_nodes if n.metadata["order"] not in small_chunk_nodes_ids]
        
        small_chunk_nodes = small_chunk_nodes + additional_small_chunk_nodes[:2]
        
        raw_text = raw_text + "\n".join([n.page_content for n in small_chunk_nodes])
        prompt = self.streaming_followup.format(followup=query, query=previous_answer["query"],
                                      answer=answer,
                                      fragment=raw_text)

        prompt = get_first_last_parts(prompt, 1000, MODEL_TOKENS_SMART - 1000)
        generator = llm(prompt, temperature=0.7, stream=True)
        answer = ''
        
        for txt in generator:
            yield txt
            answer += txt
        self.put_answer(previous_answer["query"], answer, query, mode)

    def streaming_get_more_details(self, query, answer, additional_info):
        llm = CallLLm(self.get_api_keys(), use_gpt4=True)
        prompt = prompts.get_more_details_prompt.format(query=query, answer=answer, additional_info=additional_info)
        prompt = get_first_last_parts(prompt, 1000, 6500) if llm.use_gpt4 else get_first_last_parts(prompt, 1000, 2500)
        answer = answer + "\n"
        for txt in llm(prompt, temperature=0.7, stream=True):
            yield txt
            answer += txt

    def streaming_build_summary(self):
        summary_prompt = "The given text is part of a document. Write a detailed summary which contains all important and essential information from the given text. Summarize the text:\n '{}' \nSummary: \n"
        if len(self.get_doc_data("qna_data", "chunked_summary")) > 0 and len(self.get_doc_data("qna_data", "chunked_summary")[0].strip())>0:
            # We already have the summary
            for txt in self.get_doc_data("qna_data", "chunked_summary"):
                yield txt
        running_summaries = []
        self.set_doc_data("qna_data", "chunked_summary", [])
        running_summary = ''
        this_chunk = ''
        llm = CallLLm(self.get_api_keys(), use_16k=True)
        brief_summary = self.title + "\n" + self.short_summary
        brief_summary = (brief_summary + "\n\n") if len(brief_summary.strip()) > 0 else ""
        chunks = ChunkText(self.get_doc_data("static_data", "doc_text"), TOKEN_LIMIT_FOR_DETAILED - 2000, 256)
        chunks = [f"Overall document context:\n'''{brief_summary}'''\nText from current document context we are summarising:\n'''{t}'''" for t in chunks if len(t.strip()) > 0]
        chunk_summaries = []
        for ic, chunk in enumerate(chunks):
            if not TextLengthCheck(running_summary, 1600):
                running_summaries.append(running_summary)
                running_summary = CallLLm(self.get_api_keys(), use_gpt4=False)(summary_prompt.format(running_summary), temperature=0.7, stream=False)
                
            cur_sum = f"The summary we have written till now:\n'''{running_summary}'''\nContinue writing ahead from the 'summary we have written till now'." if len(running_summary.strip()) > 0 else ""
            prev_sum = f"Summary of previous context from the same document:\n'''{this_chunk}'''" if len(this_chunk.strip()) > 0 else ""
            prompt = self.running_summary_prompt.format(summary=cur_sum, document=chunk, previous_chunk_summary=prev_sum)
            this_chunk = ''
            for txt in llm(prompt, temperature=0.7, stream=True):
                this_chunk = this_chunk + txt
                yield txt
            
            chunk_summaries.append(this_chunk)
            running_summary = running_summary + " " + this_chunk

        if len(running_summaries) == 1:
            rsum = running_summaries[0]
        elif len(running_summaries) == 0:
            rsum = running_summary
        else:
            llm = CallLLm(self.get_api_keys(), use_gpt4=True)
            if llm.use_gpt4:
                rs = [running_summaries[i] for i in range(0, len(running_summaries), 1)]
                if get_gpt4_word_count(" ".join(rs)) < 7000:
                    running_summaries = [running_summaries[i] for i in range(0, len(running_summaries), 1)]
                else:
                    rs = [running_summaries[i] for i in range(0, len(running_summaries), 2)]
                    if get_gpt4_word_count(" ".join(rs)) < 7000:
                        running_summaries = [running_summaries[i] for i in range(0, len(running_summaries), 2)]
                    else:
                        mid = max(len(running_summaries) // 2 - 1, 0)
                        running_summaries = running_summaries[mid:mid + 1]
            else:
                mid = max(len(running_summaries)//2 - 1, 0)
                running_summaries = running_summaries[mid:mid+1]
            yield '\n\n</br></br>'
            new_summary_prompt = "Write a detailed overall summary of a document from given sectional summary of parts of the document. Ignore References. \nSectional Summaries:\n'{}'\nProvide elaborate, detailed, comprehensive, informative and in-depth summary. Overall Summary:\n"
            rsum = ''
            prompt = new_summary_prompt.format(" \n".join([brief_summary] + running_summaries+[running_summary]))
            prompt = get_first_last_parts(prompt, 1000, 6000)
            yield "<h3>Overall Summary</h3>"
            yield "\n"
            for txt in llm(prompt, temperature=0.7, stream=True):
                rsum = rsum + txt
                yield txt
        
        self.set_doc_data("qna_data", "chunked_summary", chunk_summaries, overwrite=True) 
        assert len(rsum.strip()) > 0
        self.set_doc_data("qna_data", "running_summary", rsum, overwrite=True)
        self.set_doc_data("indices", "summary_index", create_index_faiss(self.get_doc_data("qna_data", "chunked_summary",), get_embedding_model(self.get_api_keys()), ))
    
    def get_instruction_text_from_review_topic(self, review_topic):
        instruction_text = ''
        if isinstance(review_topic, str) and review_topic.strip() in review_params:
            instruction_text = review_topic + ": "+review_params[review_topic.strip()]
        elif isinstance(review_topic, str):
            instruction_text = review_topic.strip()
        elif isinstance(review_topic, (list, tuple)):
            try:
                assert len(review_topic) == 2
                assert isinstance(review_topic[0], str)
                assert isinstance(review_topic[1], int)
                
                instruction_text = ": ".join(review_params[review_topic[0].strip()][review_topic[1]])
            except Exception as e:
                raise Exception(f"Invalid review topic {review_topic}")
        else:
            raise Exception(f"Invalid review topic {review_topic}")
        return instruction_text
    
    def get_all_reviews(self):
        new_review_params = dict(**review_params)
        del new_review_params["meta_review"]
        del new_review_params["scores"]
        new_reviews = []
        if self.get_doc_data("review_data"):
            for r in self.get_doc_data("review_data"):
                # dict(review_text=review_text, is_meta_review=is_meta_review, tone=tone, header=header, detailed_instructions=detailed_instructions, ) we use this structure.
                new_reviews.append(dict(review=r["review"] + ('\n\n' if len(r['score']) > 0 else '') + r['score'], 
                                        is_meta_review=r["is_meta_review"], 
                                        tone=r["tone"], 
                                        id=r["id"],
                                        review_topic=r["review_topic"],
                                        header=self.get_instruction_text_from_review_topic(r["review_topic"]).split(":")[0].strip(), 
                                        description=self.get_instruction_text_from_review_topic(r["review_topic"]).split(":")[-1].strip(),
                                        instructions=r["additional_instructions"],))    
            
            return {"reviews": new_reviews, "review_params": new_review_params}
        else:
            return {"reviews": [], "review_params":new_review_params}
       
        
    def get_review(self, tone, review_topic, additional_instructions, score_this_review, use_previous_reviews, is_meta_review):
        # Map -> collect details.
        # TODO: Support followup on a generated review.
        # TODO: use previous reviews.
        assert tone in ["positive", "negative", "neutral", "none"]
        tones = ["positive", "negative", "neutral", "none"]
        tone_synonyms = ["favorable and supportive.", "critical and unfavorable.", "not opinionated and middle grounded.", "unbiased to accept or reject decision."]
        instruction_text = self.get_instruction_text_from_review_topic(review_topic)
        if is_meta_review:
            assert use_previous_reviews and self.get_doc_data("review_data") is not None and len(self.get_doc_data("review_data")) > 0, "Meta reviews require previous reviews to be present"
        # fetch cached review if present.
        # if self.get_doc_data("review_data"):
        #     for review in self.get_doc_data("review_data"):
        #         if str(review["review_topic"]) == str(review_topic) and review["tone"] == tone:
        #             yield review["review"]
        #             yield review["score"]
        #             return
        previous_reviews_text = ''
        newline = "\n"
        if use_previous_reviews and self.get_doc_data("review_data") and len(self.get_doc_data("review_data")) > 0:
            previous_reviews = [review for review in self.get_doc_data("review_data") if review["tone"] == tone]
            previous_reviews_text = "\n\n".join([review["review"]+review["score"] for review in previous_reviews])
        query_prompt = f"""You are an expert {'meta-' if is_meta_review else ''}reviewer assigned to write an in-depth review and evaluate a scientific research paper using provided reviewer instructions on a conference submission website like openreview.net or microsoft cmt. 
Justify your review with examples from the research paper.{(' '+review_params['meta_review'] + ' ') if is_meta_review else ''} Provide a {(tone + ' ') if tone!='none' and len(tone)>0 else ''}review for the given scientific research.
{(' Make your review sound ' + tone_synonyms[tones.index(tone)]) if tone!='none' and len(tone)>0 else ''}
The topic and style you should follow while writing the review is described in the reviewer instructions given below:\n'''{instruction_text}'''.
{('Further we have certain additional instructions to follow while writing this review: ```' + additional_instructions + '```' + newline) if len(additional_instructions.strip())>0 else ''}{('We also have previous reviews with same tone on this paper to assist in writing this review. Previous reviews: ```' + previous_reviews_text + '```' + newline) if len(previous_reviews_text) > 0 else ''} 
Don't give final remarks or conclusions unless asked in reviewer instructions. 
\n{'Meta-' if is_meta_review else ''}Review: \n"""
        mode = defaultdict(lambda: False)
        mode["review"] = True
        review = ''
        for txt in self.streaming_get_short_answer(query_prompt, defaultdict(lambda: False, {"review": True}), save_answer=False):
            yield txt
            review += txt
        score = ''
        
        if score_this_review:
            score_prompt = f"""Provide a score for the given research work using the given review on a scale of 1-5 ({review_params['scores']}). 
Provide your step by step elaborate reasoning for your score decision before writing your score.
First page of the research work:  \n'''{ ' '.join(self.get_doc_data("raw_data", "chunks")[:3])}''' \nReview: \n'''{review}''' \nWrite Reasoning for score and then write score: \n"""
            for txt in CallLLm(self.get_api_keys(), use_gpt4=False)(score_prompt, temperature=0.1, stream=True):
                yield txt
                score += txt
        self.save_review(review, score, tone, review_topic, additional_instructions, is_meta_review)
    
    def save_review(self, review, score, tone, review_topic, additional_instructions, is_meta_review):
        if self.get_doc_data("review_data") is None:
            self.set_doc_data("review_data", None, [])
        
        save_dict = dict(review=review, score=score, tone=tone, review_topic=",".join(map(str, review_topic)) if isinstance(review_topic, list) else review_topic, additional_instructions=additional_instructions, is_meta_review=is_meta_review)
        id = str(mmh3.hash(self.doc_source + ",".join([tone, ",".join(map(str, review_topic)) if isinstance(review_topic, list) else review_topic, additional_instructions, str(is_meta_review)]), signed=False))
        save_dict["id"] = id
        self.set_doc_data("review_data", None, [save_dict])

        
    @staticmethod
    def load_local(folder):
        original_folder = folder
        folder = os.path.join(folder, os.path.basename(folder)+".index")
        import dill
        try:
            with open(folder, "rb") as f:
                obj = dill.load(f)
                setattr(obj, "_storage", original_folder)
                return obj
        except Exception as e:
            logger.error(f"Error loading from local storage {folder} with error {e}")
            try:
                shutil.rmtree(original_folder)
            except Exception as e:
                logger.error(
                    f"Error deleting local storage {folder} with error {e}")
            return None
    
    def save_local(self):
        import dill
        doc_id = self.doc_id
        folder = self._storage
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "locks"), exist_ok=True)
        path = Path(folder)
        lock_location = os.path.join(os.path.join(path.parent.parent, "locks"), f"{doc_id}")
        filepath = os.path.join(folder, f"{doc_id}.index")
        lock = FileLock(f"{lock_location}.lock")
        if hasattr(self, "api_keys"):
            presave_api_keys = self.api_keys
            self.api_keys = {k: None for k, v in self.api_keys.items()}
        
        with lock.acquire(timeout=600):
            previous_attr = dict()
            for k in self.store_separate:
                if hasattr(self, k):
                    previous_attr[k] = getattr(self, k)
                    setattr(self, k, None)
            with open(filepath, "wb") as f:
                dill.dump(self, f)
            for k, v in previous_attr.items():
                setattr(self, k, v)
        if hasattr(self, "api_keys"):
            self.api_keys = presave_api_keys
    

    def put_answer(self, query, answer, followup_query='', mode=None):
        query = query.strip()
        followup_query = followup_query.strip()
        final_query = query + (f". followup:{followup_query}" if len(followup_query.strip()) > 0 else "")
        question_id = str(mmh3.hash(self.doc_source + final_query, signed=False))
        found_index = None
        for ix, qna_pair in enumerate(self.get_doc_data("qna_data", "detailed_qna")):
            if qna_pair[0] == question_id and found_index is None:
                found_index = ix
        logger.info(f"Put answer in doc for storage with question_id = {question_id}, query = {query}, found_index = {found_index}")
        if found_index is None:
            self.set_doc_data("qna_data", "detailed_qna", [[question_id, final_query, answer, mode]])
        else:
            self.get_doc_data("qna_data", "detailed_qna")[found_index] = [question_id, final_query, answer, mode]
            self.set_doc_data("qna_data", "detailed_qna", self.get_doc_data("qna_data", "detailed_qna"), overwrite=True)
        db2 = FAISS.from_texts([final_query +"\n"+answer], get_embedding_model(self.get_api_keys()))
        self.get_doc_data("indices", "dqna_index").merge_from(db2)
        index = self.get_doc_data("indices", "dqna_index")
        self.set_doc_data("indices", "dqna_index", index)
        
    
    
    def get_api_keys(self):
        logger.info(f"get api keys for self hash = {hash(self)} and doc_id = {self.doc_id}")
        if hasattr(self, "api_keys"):
            api_keys = deepcopy(self.api_keys)
        else:
            raise AttributeError("No attribute named `api_keys`.")
        return api_keys
    
    
    def set_api_keys(self, api_keys:dict):
        assert isinstance(api_keys, dict)
        indices = self.get_doc_data("indices")
        for k, j in indices.items():
            if isinstance(j, (FAISS, VectorStore)):
                j.embedding_function = get_embedding_model(api_keys).embed_query
                j.embedding_function.__self__.openai_api_key = api_keys["openAIKey"]
                setattr(j.embedding_function.__self__, "openai_api_key", api_keys["openAIKey"])
        setattr(self, "api_keys", api_keys)
    
    def __copy__(self):
        # Create a new instance of our class
        cls = self.__class__
        result = cls.__new__(cls)
        # Copy all attributes from self to result. This is a shallow copy.
        result.__dict__.update(self.__dict__)
        for k in self.store_separate:
            if hasattr(result, k):
                setattr(result, k, None)
        
        if hasattr(result, "api_keys"):
            result.api_keys = deepcopy(self.api_keys)
        
        return result
    
    def copy(self):
        return self.__copy__()



class ImmediateDocIndex(DocIndex):
    pass

def create_immediate_document_index(pdf_url, folder, keys)->DocIndex:
    from langchain.document_loaders import UnstructuredMarkdownLoader
    from langchain.document_loaders import JSONLoader
    from langchain.document_loaders import UnstructuredHTMLLoader
    from langchain.document_loaders.csv_loader import CSVLoader
    from langchain.document_loaders import UnstructuredWordDocumentLoader
    from langchain.document_loaders import TextLoader
    pdf_url = pdf_url.strip()
    # check if the link is local or remote
    is_remote = pdf_url.startswith("http") or pdf_url.startswith("ftp") or pdf_url.startswith("s3") or pdf_url.startswith("gs") or pdf_url.startswith("azure") or pdf_url.startswith("https") or pdf_url.startswith("www.")
    if is_remote:
        pdf_url = convert_to_pdf_link_if_needed(pdf_url)
        is_pdf = is_pdf_link(pdf_url)
    else:
        is_pdf = pdf_url.endswith(".pdf")
    # based on extension of the pdf_url decide on the loader to use, in case no extension is present then try pdf, word, html, markdown in that order.
    logger.info(f"Creating immediate doc index for {pdf_url}, is_remote = {is_remote}, is_pdf = {is_pdf}")
    if is_pdf:
        doc_text = PDFReaderTool(keys)(pdf_url)
    elif pdf_url.endswith(".docx"):
        doc_text = UnstructuredWordDocumentLoader(pdf_url).load()[0].page_content
    elif is_remote and not (pdf_url.endswith(".md") or pdf_url.endswith(".json") or pdf_url.endswith(".csv") or pdf_url.endswith(".txt")):
        html = fetch_html(pdf_url, keys["zenrows"], keys["brightdataUrl"])
        # save this html to a file and then use the html loader.
        html_file = os.path.join(folder, "temp.html")
        with open(html_file, "w") as f:
            f.write(html)
        convert_doc_to_pdf(html_file, html_file.replace(".html", ".pdf"))
        pdf_url = html_file.replace(".html", ".pdf")
        # delete html file
        os.remove(html_file)
        doc_text = UnstructuredHTMLLoader(html_file).load()[0].page_content
    elif pdf_url.endswith(".html"):
        doc_text = UnstructuredHTMLLoader(pdf_url).load()[0].page_content
    elif pdf_url.endswith(".md"):
        doc_text = UnstructuredMarkdownLoader(pdf_url).load()[0].page_content
    elif pdf_url.endswith(".json"):
        doc_text = JSONLoader(pdf_url).load()[0].page_content
    elif pdf_url.endswith(".csv"):
        doc_text = CSVLoader(pdf_url).load()[0].page_content
    elif pdf_url.endswith(".txt"):
        doc_text = TextLoader(pdf_url).load()[0].page_content
    else:
        try:
            doc_text = PDFReaderTool(keys)(pdf_url.strip())
        except Exception as e:
            try:
                doc_text = UnstructuredWordDocumentLoader()(pdf_url.strip()).load()[0].page_content
            except Exception as e:
                try:
                    doc_text = UnstructuredHTMLLoader()(pdf_url.strip()).load()[0].page_content
                except Exception as e:
                    try:
                        doc_text = UnstructuredMarkdownLoader()(pdf_url.strip()).load()[0].page_content
                    except Exception as e:
                        raise Exception(f"Could not find a suitable loader for the given url {pdf_url}")
    
        
    doc_text = doc_text.replace('<|endoftext|>', '\n').replace('endoftext', 'end_of_text').replace('<|endoftext|>', '')
    chunks = get_async_future(ChunkText, doc_text, LARGE_CHUNK_LEN, 64)
    small_chunks = get_async_future(ChunkText, doc_text, SMALL_CHUNK_LEN, 32)
    chunks, small_chunks = chunks.result(), small_chunks.result()
    nested_dict = {
        'chunked_summary': [''],
        'chunks': chunks,
        "small_chunks": small_chunks,
        'running_summary': '',
        'detailed_qna': [],
        'deep_reader_details': {
            "methodology": {"id":"", "text":""},
            "previous_literature_and_differentiation": {"id":"", "text":""},
            "experiments_and_evaluation": {"id":"", "text":""},
            "results_and_comparison": {"id":"", "text":""},
            "limitations_and_future_work" : {"id":"", "text":""},
        }
    }
    openai_embed = get_embedding_model(keys)
    try:
        doc_index: DocIndex = ImmediateDocIndex(pdf_url,
                    "pdf", 
                    "scientific_article", doc_text, nested_dict, openai_embed, folder)
        # for k in doc_index.store_separate:
        #     doc_index.set_doc_data(k, None, doc_index.get_doc_data(k), overwrite=True)
        def get_doc_ready():
            doc_index.set_api_keys(keys)
            return doc_index.get_short_info()
        _ = get_async_future(get_doc_ready)
        doc_index._visible = True
    except Exception as e:
        doc_id = str(mmh3.hash(pdf_url + "pdf" + "scientific_article", signed=False))
        try:
            folder = os.path.join(folder, f"{doc_id}")
            if os.path.exists(folder):
                shutil.rmtree(folder)
        except Exception as e:
            pass
        logger.error(f"Error creating immediate doc index for {pdf_url}")
        raise e
    
    return doc_index

