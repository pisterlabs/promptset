import asyncio
from typing import List, Optional

from aiofauna.llm.llm import *
from aiofauna.llm.llm import LLMStack, UpsertRequest, UpsertVector
from aiofauna.typedefs import FunctionType
from aiofauna.utils import handle_errors, setup_logging
from aiohttp import ClientSession, TCPConnector
from bs4 import BeautifulSoup
from langchain.embeddings import OpenAIEmbeddings
from pydantic import Field, HttpUrl

from ..config import env

openai_embeddings = OpenAIEmbeddings()  # type: ignore

PINECONE_URL = env.PINECONE_API_URL
PINECONE_API_KEY = env.PINECONE_API_KEY
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
}
BAD_EXT = (
    "png",
    "jpg",
    "jpeg",
    "gif",
    "pdf",
    "doc",
    "docx",
    "ppt",
    "pptx",
    "xls",
    "xlsx",
    "zip",
    "rar",
    "gz",
    "7z",
    "exe",
    "mp3",
    "mp4",
    "avi",
    "mkv",
    "mov",
    "wmv",
    "flv",
    "swf",
)

connector = TCPConnector(limit=500)

logger = setup_logging(__name__)


@handle_errors
async def sitemap(url: str, session: ClientSession) -> List[str]:
    urls = []
    if not url.endswith("xml"):
        url = f"{url.rstrip('/')}/sitemap.xml"
    async with session.get(url) as response:
        text = await response.text()
        soup = BeautifulSoup(text, features="xml")
        for loc in soup.findAll("loc"):
            if loc.text.endswith(BAD_EXT):
                continue
            urls.append(loc.text)
            logger.info(f"Found {loc.text}")
        for nested_sitemap in soup.findAll("sitemap"):
            urls.extend(await sitemap(nested_sitemap.loc.text, session))
    return urls


@handle_errors
async def upsert_embeddings(texts: List[str], namespace: str, session: ClientSession):
    embeddings = await openai_embeddings.aembed_documents(texts)
    vectors = UpsertRequest(
        namespace=namespace,
        vectors=[
            UpsertVector(id=str(uuid4()), values=vector, metadata={"text": text})
            for text, vector in zip(texts, embeddings)
        ],
    )
    async with session.post("/vectors/upsert", json=vectors._asdict()) as response:
        if response.status != 200:
            raise RuntimeError(await response.text())


@handle_errors
async def fetch_website(url: str, session: ClientSession, max_size: int = 40960) -> str:
    async with session.get(url) as response:
        html = await response.text()
        truncated_html = html[:max_size]
        return BeautifulSoup(truncated_html, features="lxml").get_text(
            separator="\n", strip=True
        )


async def sitemap_pipeline(
    url: str,
    namespace: str,
    session: ClientSession,
    pinecone_session: ClientSession,
    chunk_size: int = 32,
):
    urls = await sitemap(url, session)
    length = len(urls)
    inserted = 0
    while urls:
        chunk = urls[:chunk_size]
        urls = urls[chunk_size:]
        try:
            contents = await asyncio.gather(
                *[fetch_website(url, session) for url in chunk]
            )
            await upsert_embeddings(contents, namespace, pinecone_session)
            inserted += len(chunk)
            progress = inserted / length
            logger.info(f"Progress: {progress}")
            if progress >= 1:
                yield "100"
                break
            yield str(progress * 100)
        except Exception as e:
            logger.error(e)
            continue


llm = LLMStack(base_url=PINECONE_URL, headers={"api-key": PINECONE_API_KEY})


class IngestSiteMap(FunctionType):
    """Iterates over all the nested sitemap.xml related files, finds the loc of the content urls,
    scraps those urls and upserts the embeddings of the page content into the pinecone vector store.
    for further similarity search in order to feed a Large Language Model (LLM) with the content as a
    knowledge base.
    """

    url: HttpUrl = Field(..., description="The base url of the website to be ingested")
    namespace: str = Field(
        ...,
        description="The name of the pinecone vector store where the embeddings will be stored",
    )

    @handle_errors
    async def run(self):
        async with ClientSession(headers=HEADERS) as session:
            progress = []
            async with ClientSession(
                base_url=env.PINECONE_API_URL, headers={"api-key": env.PINECONE_API_KEY}
            ) as pinecone:
                async for data in sitemap_pipeline(
                    url=self.url,
                    namespace=self.namespace,
                    session=session,
                    pinecone_session=pinecone,
                ):
                    progress.append(data)
                    if data == "100":
                        break
