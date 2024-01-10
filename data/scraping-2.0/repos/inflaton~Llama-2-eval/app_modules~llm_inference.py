import abc
import os
import time
import urllib
from queue import Queue
from threading import Thread
from typing import List, Optional
from urllib.parse import quote, urlparse, urlunparse

from langchain.chains.base import Chain

from app_modules.llm_loader import LLMLoader, TextIteratorStreamer
from app_modules.utils import remove_extra_spaces


class LLMInference(metaclass=abc.ABCMeta):
    llm_loader: LLMLoader
    chain: Chain

    def __init__(self, llm_loader):
        self.llm_loader = llm_loader
        self.chain = None

    @abc.abstractmethod
    def create_chain(self, inputs) -> Chain:
        pass

    def get_chain(self, inputs) -> Chain:
        if self.chain is None:
            self.chain = self.create_chain(inputs)

        return self.chain

    def run_chain(self, chain, inputs, callbacks: Optional[List] = []):
        return chain(inputs, callbacks)

    def call_chain(
        self,
        inputs,
        streaming_handler,
        q: Queue = None,
        testing: bool = False,
    ):
        print(inputs)
        if self.llm_loader.streamer.for_huggingface:
            self.llm_loader.lock.acquire()

        try:
            self.llm_loader.streamer.reset(q)

            chain = self.get_chain(inputs)
            result = (
                self._run_chain_with_streaming_handler(
                    chain, inputs, streaming_handler, testing
                )
                if streaming_handler is not None
                else self.run_chain(chain, inputs)
            )

            if "answer" in result:
                result["answer"] = remove_extra_spaces(result["answer"])

                source_path = os.environ.get("SOURCE_PATH")
                base_url = os.environ.get("PDF_FILE_BASE_URL")
                if base_url is not None and len(base_url) > 0:
                    documents = result["source_documents"]
                    for doc in documents:
                        source = doc.metadata["source"]
                        title = source.split("/")[-1]
                        doc.metadata["url"] = f"{base_url}{urllib.parse.quote(title)}"
                elif source_path is not None and len(source_path) > 0:
                    documents = result["source_documents"]
                    for doc in documents:
                        source = doc.metadata["source"]
                        url = source.replace(source_path, "https://")
                        url = url.replace(".html", "")
                        parsed_url = urlparse(url)

                        # Encode path, query, and fragment
                        encoded_path = quote(parsed_url.path)
                        encoded_query = quote(parsed_url.query)
                        encoded_fragment = quote(parsed_url.fragment)

                        # Construct the encoded URL
                        doc.metadata["url"] = urlunparse(
                            (
                                parsed_url.scheme,
                                parsed_url.netloc,
                                encoded_path,
                                parsed_url.params,
                                encoded_query,
                                encoded_fragment,
                            )
                        )

            return result
        finally:
            if self.llm_loader.streamer.for_huggingface:
                self.llm_loader.lock.release()

    def _execute_chain(self, chain, inputs, q, sh):
        q.put(self.run_chain(chain, inputs, callbacks=[sh]))

    def _run_chain_with_streaming_handler(
        self, chain, inputs, streaming_handler, testing
    ):
        que = Queue()

        t = Thread(
            target=self._execute_chain,
            args=(chain, inputs, que, streaming_handler),
        )
        t.start()

        if self.llm_loader.streamer.for_huggingface:
            count = (
                2
                if "chat_history" in inputs and len(inputs.get("chat_history")) > 0
                else 1
            )

            while count > 0:
                try:
                    for token in self.llm_loader.streamer:
                        if not testing:
                            streaming_handler.on_llm_new_token(token)

                    self.llm_loader.streamer.reset()
                    count -= 1
                except Exception:
                    if not testing:
                        print("nothing generated yet - retry in 0.5s")
                    time.sleep(0.5)

        t.join()
        return que.get()
