import json
from pathlib import Path
from typing import Dict, Generator, List

import requests
import typer

from .cache import Cache
from .config import cfg
# for embeddings
import os
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

CACHE_LENGTH = int(cfg.get("CACHE_LENGTH"))
CACHE_PATH = Path(cfg.get("CACHE_PATH"))
REQUEST_TIMEOUT = int(cfg.get("REQUEST_TIMEOUT"))
DISABLE_STREAMING = str(cfg.get("DISABLE_STREAMING"))

HTTP_REFERER = str(cfg.get("HTTP_REFERER"))
APP_TITLE = str(cfg.get("APP_TITLE"))

EMBEDDING_TEMPLATE = """Answer the question based on the following context:

{context}

Question: {question}
"""

class OpenAIClient:
    cache = Cache(CACHE_LENGTH, CACHE_PATH)

    def __init__(self, api_host: str, api_key: str) -> None:
        self.__api_key = api_key
        self.api_host = api_host

    def format_docs(self, docs):
        """
        Format the documents for context retrieval.
        :param docs: List of documents.
        :return: String containing all documents.
        """
        return "\n\n".join([d.page_content for d in docs])

    def load_vsindex(self, directory_path):
        """
        Load a whole directory into a vector store index. If a '.persist' directory 
        exists within the given directory, load the index from there. Otherwise, 
        create a new index, store it in '.persist', and return it.

        :param directory_path: Path to the directory containing documents to index.
        :return: The loaded or created vector store index.
        """
        persist_path = os.path.join(directory_path, ".persist")

        if os.path.exists(persist_path):
            vectorstore = Chroma(persist_directory=persist_path, embedding_function=OpenAIEmbeddings())
            index = VectorStoreIndexWrapper(vectorstore=vectorstore)
        else:
            os.makedirs(persist_path, exist_ok=True)
            loader = DirectoryLoader(directory_path)
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": persist_path}).from_loaders([loader])
        return index

    def add_context(self, local_path, messages):
        """
        Add context to the last message in the list of messages.
        :param local_path: Path to the directory containing documents to index.
        :param messages: List of dict with messages and roles.
        :return: None.
        """
        question = messages[-1]["content"]
        vectorstore = self.load_vsindex(local_path).vectorstore
        embedding_vector = OpenAIEmbeddings().embed_query(question)
        docs = vectorstore.similarity_search_by_vector(embedding_vector, 4)
        context = self.format_docs(docs)
        messages[-1]["content"] = EMBEDDING_TEMPLATE.format(context=context, question=question)

    @cache
    def _request(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 1,
        top_probability: float = 1,
    ) -> Generator[str, None, None]:
        """
        Make request to OpenAI API, read more:
        https://platform.openai.com/docs/api-reference/chat

        :param messages: List of messages {"role": user or assistant, "content": message_string}
        :param model: String gpt-3.5-turbo or gpt-3.5-turbo-0301
        :param temperature: Float in 0.0 - 2.0 range.
        :param top_probability: Float in 0.0 - 1.0 range.
        :return: Response body JSON.
        """
        # embeddings
        self.add_context(os.getcwd(), messages)

        stream = DISABLE_STREAMING == "false"
        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "top_p": top_probability,
            "stream": stream,
        }
        endpoint = f"{self.api_host}/v1/chat/completions"
        response = requests.post(
            endpoint,
            # Hide API key from Rich traceback.
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.__api_key}",
                "HTTP-Referer": HTTP_REFERER,
                "X-Title": APP_TITLE,
            },
            json=data,
            timeout=REQUEST_TIMEOUT,
            stream=stream,
        )
        response.raise_for_status()
        # TODO: Optimise.
        # https://github.com/openai/openai-python/blob/237448dc072a2c062698da3f9f512fae38300c1c/openai/api_requestor.py#L98
        if not stream:
            data = response.json()
            yield data["choices"][0]["message"]["content"]  # type: ignore
            return
        for line in response.iter_lines():
            # openrouter first line
            if line.startswith(b": OPENROUTER PROCESSING"):
                continue
            data = line.lstrip(b"data: ").decode("utf-8")
            if data == "[DONE]":  # type: ignore
                break
            if not data:
                continue
            data = json.loads(data)  # type: ignore
            delta = data["choices"][0]["delta"]  # type: ignore
            if "content" not in delta:
                continue
            yield delta["content"]

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 1,
        top_probability: float = 1,
        caching: bool = True,
    ) -> Generator[str, None, None]:
        """
        Generates single completion for prompt (message).

        :param messages: List of dict with messages and roles.
        :param model: String gpt-3.5-turbo or gpt-3.5-turbo-0301.
        :param temperature: Float in 0.0 - 1.0 range.
        :param top_probability: Float in 0.0 - 1.0 range.
        :param caching: Boolean value to enable/disable caching.
        :return: String generated completion.
        """
        yield from self._request(
            messages,
            model,
            temperature,
            top_probability,
            caching=caching,
        )
