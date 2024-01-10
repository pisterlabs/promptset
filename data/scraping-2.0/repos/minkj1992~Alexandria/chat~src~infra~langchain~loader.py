import asyncio
from typing import Callable, List, Optional, Set, Union

import httpx
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger


class AsyncRecursiveUrlLoader(RecursiveUrlLoader):
    def __init__(
        self,
        url: str,
        max_depth: Optional[int] = None,
        use_async: Optional[bool] = None,
        extractor: Optional[Callable[[str], str]] = None,
        exclude_dirs: Optional[str] = None,
        timeout: Optional[int] = None,
        prevent_outside: Optional[bool] = None,
        semaphore_count: int = 10,
    ):
        super().__init__(
            url=url,
            max_depth=max_depth,
            use_async=use_async,
            extractor=extractor,
            exclude_dirs=exclude_dirs,
            timeout=timeout,
            prevent_outside=prevent_outside,
        )
        self.semaphore = asyncio.Semaphore(
            semaphore_count
        )  # Limit the number of concurrent tasks

    async def aload(self) -> List[Document]:
        """Load web pages asynchronously."""
        if self._should_skip_url(self.url, 0):
            return []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            root_document = await self._process_link(client, self.url)

        visited = set([self.url])
        child_documents = (
            await self._async_get_child_links_recursive(self.url, visited) or []
        )

        return [doc for doc in ([root_document] + child_documents) if doc is not None]

    async def _async_get_child_links_recursive(
        self, url: str, visited: Optional[Set[str]] = None, depth: int = 0
    ) -> List[Document]:
        logger.info(f"Recursive crawl url {url}")

        if self._should_skip_url(url, depth):
            return []

        visited = set() if visited is None else visited

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            text = await self._fetch_url_content(client, url)
            if not text:
                return []

            absolute_paths = self._get_sub_links(text, url)
            documents = await self._process_links(client, absolute_paths, visited)
            sub_documents = await self._fetch_sub_documents(
                client, absolute_paths, visited, depth
            )
            return documents + sub_documents

    def _should_skip_url(self, url: str, depth: int) -> bool:
        if depth >= self.max_depth:
            return True
        if self.exclude_dirs:
            return any(url.startswith(exclude_dir) for exclude_dir in self.exclude_dirs)
        return False

    async def _fetch_url_content(self, client, url: str) -> Optional[str]:
        async with self.semaphore:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
            except (httpx.InvalidURL, Exception) as e:
                logger.error(f"Error fetching {url}: {e}")
                return None

    async def _process_links(
        self, client, links: List[str], visited: Set[str]
    ) -> List[Document]:
        tasks = [
            self._process_link(client, link) for link in links if link not in visited
        ]
        visited.update(links)
        return list(filter(None, await asyncio.gather(*tasks)))

    async def _process_link(self, client, link: str) -> Optional[Document]:
        async with self.semaphore:
            try:
                response = await client.get(link)
                response.raise_for_status()

                text = response.text
                extracted = self.extractor(text)
                if extracted:
                    return Document(
                        page_content=extracted, metadata=self._gen_metadata(text, link)
                    )
            except (httpx.InvalidURL, Exception) as e:
                logger.error(f"Error processing {link}: {e}")
                return None

    async def _fetch_sub_documents(
        self, client, links: List[str], visited: Set[str], depth: int
    ) -> List[Document]:
        tasks = [
            self._async_get_child_links_recursive(link, visited, depth + 1)
            for link in links
        ]
        return [
            doc for sublist in await asyncio.gather(*tasks) for doc in sublist if doc
        ]


async def load_url(url: str, max_depth: int) -> List[Document]:
    logger.info(f"Crawl url {url}")
    loader = AsyncRecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
        extractor=lambda x: BeautifulSoup(x, "lxml").text,
        prevent_outside=True,
    )
    temp_docs = await loader.aload()
    return [doc for i, doc in enumerate(temp_docs) if doc not in temp_docs[:i]]


async def get_docs_from_urls(urls: List[str], max_depth: int) -> List[Document]:
    # Use asyncio.gather to load all URLs concurrently
    documents = await asyncio.gather(*(load_url(url, max_depth) for url in urls))
    documents = [doc for sublist in documents for doc in sublist]  # Flatten the list

    html2text = Html2TextTransformer()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    docs_transformed = html2text.transform_documents(documents)
    docs_transformed = text_splitter.split_documents(docs_transformed)

    # Try to return 'source' and 'title' metadata when querying vector store
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""
    return docs_transformed
