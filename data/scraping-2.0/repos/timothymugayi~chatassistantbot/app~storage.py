import os
import pytz
import requests

from datetime import datetime
from typing import Union, List, Optional, Set
from urllib.parse import urlparse
from abc import ABC, abstractmethod

from bs4 import BeautifulSoup
from langchain import OpenAI
from llama_index import (
    download_loader,
    PromptHelper,
    LLMPredictor,
    ServiceContext,
    GPTListIndex
)

from app.configs import settings
from app.utils import timeit, remove_file


class BaseDataSource(ABC):

    @abstractmethod
    def index(self):
        """
        Returns the index vectorized index to search for answers.

        Returns:
            The index used by the Langchain.
        """

    @abstractmethod
    def get_cached_index(self, name: str) -> Union[str, None]:
        """
        Returns a cached index for the specified name, or None if the index is not cached.

        Args:
            name: The name of the cached index to retrieve.

        Returns:
            A string containing the cached index data, or None if the index is not cached.
        """


class WebDataExtractor(BaseDataSource):

    def __init__(self, root_url: str, urls: List[str], index_file: Optional[str] = "data.json") -> None:
        """
        WebDataExtractor Constructor initialize class
        Args:
            root_url: Root url of site to traverse
            urls: fallback urls if no sub child urls in parent root url found
            index_file: name of index file
        """
        if not root_url:
            raise ValueError("root url should be defined")
        if not urls:
            raise ValueError("default urls should be defined")
        child_urls = self.__collect_urls(root_url)
        if not child_urls:
            print("Root level url didnt find child urls defaulting to specific urls")
            urls.append(root_url)
            self.urls = urls
        else:
            self.urls = child_urls
        self.index_file = index_file

    def __collect_urls(self, root_url: str, visited: Set = None, max_urls: int = 20) -> Set:
        # Create an empty set to store the collected URLs
        if visited is None:
            visited = set()
        urls = set()
        # Extract the domain name from the root URL
        domain = urlparse(root_url).netloc

        # Make a request to the root URL and parse the HTML content
        try:
            response = requests.get(root_url)
        except requests.exceptions.RequestException:
            # If the request raises an exception, just return an empty set
            return urls

        soup = BeautifulSoup(response.content, "html.parser")
        # Find all the links on the root page and add them to the URL set
        for link in soup.find_all("a"):
            url = link.get("href")
            if url:
                parsed_url = urlparse(url)

                # Check if the child URL has the same domain as the root URL
                if parsed_url.netloc == domain:
                    urls.add(url)

                    # Check if we have reached the maximum number of URLs
                    if len(urls) >= max_urls:
                        return urls

        # Recursively traverse all child URLs and add them to the URL set
        for url in urls.copy():
            if url not in visited:
                print(url)
                visited.add(url)
                urls |= self.__collect_urls(url, visited, max_urls)
                # Check if we have reached the maximum number of URLs
                if len(urls) >= max_urls:
                    break

        return urls

    @timeit
    def index(self):
        prompt_helper = PromptHelper(
            settings.PROMPT_MAX_INPUT_SIZE,
            settings.PROMPT_NUM_OUTPUTS,
            settings.PROMPT_MAX_CHUNK_OVERLAP,
            chunk_size_limit=settings.PROMPT_CHUNK_SIZE_LIMIT
        )
        llm_predictor = LLMPredictor(
            llm=OpenAI(
                model_name=settings.OPENAI_MODEL.name,
                temperature=settings.OPENAI_MODEL.temperature,
                max_tokens=settings.PROMPT_NUM_OUTPUTS
            )
        )
        context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        # check if theres already an indexed file
        if not self.get_cached_index(self.index_file):
            ReadabilityWebPageReader = download_loader("ReadabilityWebPageReader")
            loader = ReadabilityWebPageReader()
            documents = []
            for url in self.urls:
                docs = loader.load_data(url=url)
                documents.extend(docs)
            # previous llama_index versions the way we initialize any vector index class was via
            # the constructor this as recently change as of version 0.5.10
            index = GPTListIndex.from_documents(documents, service_context=context)
            index.save_to_disk(self.index_file)
            return index
        else:
            index = GPTListIndex.load_from_disk(self.index_file, service_context=context)
            return index

    def remove_index_cache(self):
        remove_file(self.index_file)

    def get_cached_index(self, name: str) -> Union[str, None]:
        """
        Returns the filepath if it exists and its modification time is within the last day configured by
        settings.KNOWLEDGE_INDEX_CACHE_DURATION. If the file is outdated
        by more than one day, it is deleted and None is returned.

        Args:
            name (str): The path to the file to check.

        Returns:
            Union[str, None]: The filepath if it exists and its modification time is within the last day. None if the
            file is outdated and has been deleted.

        Raises:
            None
        """
        local_tz = pytz.timezone(settings.TIME_ZONE)
        if os.path.exists(name):
            mod_time = datetime.fromtimestamp(os.path.getmtime(name), tz=local_tz)
            if (datetime.now(tz=local_tz) - mod_time).days >= settings.KNOWLEDGE_INDEX_CACHE_DURATION:
                try:
                    os.remove(name)
                except OSError:
                    pass
                return None
            else:
                return name
        else:
            return None
