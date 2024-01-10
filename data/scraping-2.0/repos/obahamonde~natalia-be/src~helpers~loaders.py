import asyncio
import functools
import io
import time
from pathlib import Path
from typing import (Any, AsyncGenerator, Awaitable, Callable, Dict, List,
                    Optional, Tuple, TypeVar, Union)
from uuid import uuid4

from aiofauna.llm.llm import UpsertRequest, UpsertVector
from aiofauna.utils import chunker, handle_errors, setup_logging
from aiohttp import ClientSession
from aiohttp.web import FileField, Request
from langchain.embeddings import OpenAIEmbeddings
from pypdf import PdfReader

from ..config import env
from ..utils import sanitize_vector

logger = setup_logging(__name__)

openai_embeddings = OpenAIEmbeddings()  # type: ignore

async def pdf_reader(request: Request) -> AsyncGenerator[str, None]:
    """Reads a PDF file from the request and returns a list of strings"""
    data = await request.post()
    file = data["file"]
    assert isinstance(file, FileField)
    with io.BytesIO(file.file.read()) as f:
        reader = PdfReader(f)
        for page in chunker(reader.pages, 100):
            for chunk in page:
                yield chunk.extract_text()

async def ingest_pdf(texts:List[str],namespace:str):
    embeddings = await openai_embeddings.aembed_documents(texts)
    async with ClientSession(
        base_url=env.PINECONE_API_URL,
        headers={"api-key": env.PINECONE_API_KEY},
    ) as session:
        counter = 0
        for chunk in chunker(embeddings, 100):
            vectors = [UpsertVector(
                id=str(uuid4()),
                values=sanitize_vector(vector),
                metadata={"text": text}
            ) for text,vector in zip(texts, chunk)]
            request = UpsertRequest(
                namespace=namespace,
                vectors=vectors
            )
            async with session.post("/vectors/upsert", json=request.dict()) as response:
                if response.status != 200:
                    raise RuntimeError(await response.text())
            counter += len(chunk)
            logger.info(f"Progress: {counter/len(texts)}")
            yield str(counter/len(texts)*100)
            await asyncio.sleep(0.25)