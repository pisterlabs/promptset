import asyncio
import io
import logging
from typing import Annotated, List
from fastapi import APIRouter, Header
from langchain.text_splitter import CharacterTextSplitter

from config import Config
from models.extension import ExtensionDocument, ExtensionPDFDocument, UrlMetadataInfo
from services.bookmark_store_service import AsyncBookmarkStoreService
from services.context_service import ContextService
from utils.db import get_vectorstore, firebase_app as db
import PyPDF2

router = APIRouter()
config = Config()
log = logging.getLogger(__name__)


@router.post('/store')
async def store(document: ExtensionDocument, x_uid: Annotated[str, Header()]):
    vectorstore = get_vectorstore()
    user_id = x_uid
    chunks = CharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, separator='.'
    ).split_text(document.raw_text)
    log.info(f'created {len(chunks)} chunks')
    bookmark_service = AsyncBookmarkStoreService()
    bookmark_ref = await bookmark_service.add_bookmark(user_id, document)

    try:
        with vectorstore.batch() as batch:
            for chunk in chunks:
                batch.add_data_object({
                    "title": document.title,
                    "content": chunk,
                    "user_id": user_id,
                    "url": document.url,
                    "firebase_id": bookmark_ref.id,  # all chunks have same firebase id
                }, "Document")
            batch.flush()
    except Exception as e:
        log.error(e)
        await bookmark_service.delete_user_bookmark(user_id, document)
        return {'success': False, 'error': str(e)}

    return {'success': True}


@router.post('/storepdf')
async def store_pdf(document: ExtensionPDFDocument, x_uid: Annotated[str, Header()]):
    vectorstore = get_vectorstore()
    user_id = x_uid
    pdf_bytes = bytes(document.pdf_bytes)

    pdf_text = ''
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    num_pages = len(reader.pages)

    for page_number in range(num_pages):
        page = reader.pages[page_number]
        pdf_text += page.extract_text()

    chunks = CharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, separator='.'
    ).split_text(pdf_text)
    log.info(f'created {len(chunks)} chunks')
    bookmark_store_service = AsyncBookmarkStoreService()
    bookmark_ref = await bookmark_store_service.add_bookmark(user_id, document)

    try:
        with vectorstore.batch() as batch:
            for chunk in chunks:
                batch.add_data_object({
                    "title": document.title,
                    "content": chunk,
                    "user_id": user_id,
                    "url": document.url,
                    "firebase_id": bookmark_ref.id,  # all chunks have same firebase id
                }, "Document")
            batch.flush()
    except Exception as e:
        log.error(e)
        await bookmark_store_service.delete_user_bookmark(user_id, document)
        return {'success': False, 'error': str(e)}

    return {'success': True}


@router.get('/info')
async def url_metadata(url: str, x_uid: Annotated[str, Header()]) -> UrlMetadataInfo:
    service = AsyncBookmarkStoreService()
    bookmarks, folders = await asyncio.gather(service.get_bookmarks_by_url(x_uid, url), service.get_user_folders(x_uid))
    return UrlMetadataInfo(
        is_bookmarked=len(bookmarks) > 0,
        folders=folders,
    )

@router.post('/batch-delete')
async def batch_delete(documents: List[str], x_uid: Annotated[str, Header()], folders: List[str] = None):
    bookmark_service = AsyncBookmarkStoreService()
    context_service = ContextService(get_vectorstore())
    try:
        context_service.batch_delete(x_uid, documents)
        await bookmark_service.batch_delete(x_uid, documents, folders)
        return {'success': True}
    except Exception as e:
        log.error(e)
        return {'success': False, 'error': str(e)}