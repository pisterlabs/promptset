from langchain.vectorstores import FAISS
from datetime import datetime
from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
)
import os
from typing import Any, Dict, List, Optional, Tuple
import json
import requests
from langchain.schema import BaseMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import get_buffer_string
from langchain.memory.utils import get_prompt_input_key
from pydantic import Field
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index.langchain_helpers.text_splitter import (
    TokenTextSplitter,
)
from langchain.docstore.in_memory import InMemoryDocstore

import numpy as np
import pandas as pd
import faiss
from typing import Any, Dict, List, NamedTuple
import pickle
from langchain.agents import load_tools, initialize_agent, Tool


def mkdir_if_dne(path):
    if os.path.isdir(path):
        return False
    else:
        os.mkdir(path)
        print(f"dir created : {path}")
        return True


class FAISSChatHistory:

    """
    This Chat history loads 2*N conversation pairs, where N is max_k parameter
    First N chats:
    similarity searched conversations / nmr based similarity seached conversations
    Last N chats:
    last N chats that occured.

    """

    def __init__(self, save_dir, max_k=3, use_nmr=False, similarity_cutoff=0.5):
        self.save_dir = save_dir + "/" + "faiss_chat_memory"
        self.use_nmr = use_nmr
        self.similarity_cutoff = similarity_cutoff
        # self.mdb_client = mdb_client
        # self.coll = c_name
        # self.index = self._get_index()
        self.max_k = max_k
        self.embed_model = OpenAIEmbeddings()
        self.splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=16)
        self.docstore = InMemoryDocstore({})
        self.index = faiss.IndexFlatL2(1536)
        self.vecstore = None

        dir_bool = mkdir_if_dne(self.save_dir)
        self._init()
        if dir_bool:
            print(f"Init new Memory to path {self.save_dir}")
            return
        else:
            if os.path.isfile(self.save_dir + "/index.faiss"):
                print(f"Reload Memory from path {self.save_dir}")
                self._reload()
                print("loaded")
            else:
                print(f"Init new Memory to path {self.save_dir}")
            return

    def _init(self):
        self.vecstore = FAISS(
            self.embed_model.embed_query, self.index, self.docstore, {}
        )

    def _reload(self):
        if self.vecstore is not None:
            print("Loaded previous conversation")
            self.vecstore = FAISS.load_local(self.save_dir, self.embed_model)
            print(self.vecstore.index.ntotal)
        else:
            print("Bug hit")

    def _save(self):
        if self.vecstore is not None:
            print("saving conversation")
            self.vecstore.save_local(self.save_dir)
            print(self.vecstore.index.ntotal)

    def get_embedding(self, message: str):
        return self.embed_model.embed_query(message)

    def save_message(self, message):
        time_now = datetime.now().strftime("%b-%d-%Y || %H:%M:%S")
        extra_info = {"date-time": time_now}
        self.vecstore.add_texts([message], [extra_info])
        self._save()

    def get_index_size(self):
        return self.vecstore.index.ntotal

    def semantic_search(self, query):
        pass

    def search(self, query_str: str) -> List[str]:
        # search k conversations
        if query_str.strip() == "":
            raise Exception("Invalid query :''  ")
            return []
        query_embedding = self.get_embedding(query_str)

        cur_index_size = self.get_index_size()
        print(f"current index size {cur_index_size}, max_k : {self.max_k}")
        # when number of required N messages are not twice than what is specified, we will provide all prev convos
        if self.max_k >= 2 * cur_index_size:
            print(f"{self.max_k}>= {2*cur_index_size}")
            fetch_total = cur_index_size
            last_n = self.get_last_k_messages(k=fetch_total)
            all_n = last_n

        else:
            print(f"{self.max_k}< {2*cur_index_size}")
            if self.use_nmr:
                first_n = self.vecstore.max_marginal_relevance_search_by_vector(
                    query_embedding, k=self.max_k, fetch_k=4 * self.max_k
                )
            else:
                docs = self.vecstore.similarity_search_with_score_by_vector(
                    query_embedding, k=self.max_k
                )
                print(f"First n docs:{len(docs)}")
                first_n = []
                for e, doc in enumerate(docs):
                    # print(doc)
                    i, j = doc
                    if j > self.similarity_cutoff:
                        first_n.append(i)
            print(f"First n:{len(first_n)}")
            first_n = {i.metadata["date-time"]: i for i in first_n if not i == []}

            last_n = self.get_last_k_messages(k=self.max_k)
            last_n_ids = [i.metadata["date-time"] for i in last_n]
            intersection = set(first_n.keys()).intersection(set(last_n_ids))
            for j in intersection:
                first_n.pop(j)
            all_n = list(first_n.values()) + last_n

        return all_n

    def get_last_k_messages(self, k: int):
        last_k = []
        if k >= self.get_index_size():
            ids = list(self.vecstore.index_to_docstore_id.values())
        else:
            ids = list(self.vecstore.index_to_docstore_id.values())[-k:]

        for i in ids:
            doc = self.vecstore.docstore.search(i)
            last_k.append(doc)

        return last_k

    def _clear(self):
        self._init()
        self._save()


class FAISSChatMemory(BaseMemory):
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    memory_key = "chat_history"
    return_messages = True
    chat_memory = ChatMessageHistory()
    chat_history_model: FAISSChatHistory = None
    seperator: str = "/#/"
    max_k: int = 3

    def init(self, save_dir, max_k=3, use_nmr=False):
        self.chat_history_model = FAISSChatHistory(
            save_dir, max_k=max_k, use_nmr=use_nmr
        )
        return self

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_text = inputs[list(inputs.keys())[0]]
        result_docs = self.chat_history_model.search(input_text)
        print(f"result docs: {len(result_docs)}")
        self.chat_memory.clear()
        if result_docs == []:
            self.chat_memory.add_ai_message("No conversations so far.")
        else:
            for docs in result_docs:
                text = docs.page_content
                text = text.split(self.seperator)

                for t in text:
                    # print(t)
                    msg = json.loads(t)
                    if msg["role"] == "user":
                        full_text = (
                            msg["content"]
                            + f" [Date and Time: {docs.metadata['date-time']}"
                        )
                        self.chat_memory.add_user_message(full_text)
                    elif msg["role"] == "assistant":
                        full_text = (
                            msg["content"]
                            + f" [Date and Time: {docs.metadata['date-time']}"
                        )
                        self.chat_memory.add_ai_message(full_text)

        if self.return_messages:
            return {self.memory_key: self.chat_memory.messages}
        else:
            return {self.memory_key: get_buffer_string(self.chat_memory.messages)}

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        new_output = {"output": outputs["output"]}
        input_str, output_str = self._get_input_output(inputs, new_output)
        input_str = json.dumps({"role": "user", "content": f"{input_str}"})
        output_str = json.dumps({"role": "assistant", "content": f"{output_str}"})
        message = self.seperator.join([input_str, output_str])
        self.chat_history_model.save_message(message)

    def clear(self):
        print("Clearing previous saved conversations")
        self.chat_history_model._clear()
        self.chat_memory.clear()
