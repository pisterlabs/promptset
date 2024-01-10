"""This defines the Confluence API routes."""

import logging
import sys

from fastapi import APIRouter, Depends
from langchain.schema.vectorstore import VectorStore

from pulse.loaders import confluence_ingestor
from pulse import vectorstore


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger("pulse.routers.slack")

v1 = APIRouter()

@v1.post("/ingest")
async def ingest(vectorstore: VectorStore = Depends(vectorstore.get_vectorstore)):
    """Trigger to ingest all labelled pages from confluence"""
    confluence_ingestor(vectorstore)