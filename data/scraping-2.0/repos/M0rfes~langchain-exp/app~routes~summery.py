import uuid
import os
from typing import Annotated, Optional
import aiofiles
from fastapi import APIRouter, UploadFile
from app.llm.summery import summery, summery_from_text
from app.model.chain_type import Chain_Type
from app.model.summery import Summery_Text, Summery_URL
from langchain.document_loaders import UnstructuredURLLoader

router = APIRouter(
    prefix='/summery',
    tags=['summery']
)
OUT_FILE_DIR = "app/static"


@router.post("/")
async def summaries_pdf(file: UploadFile):
    """

    summarizes the pdf with provided chain type
    chain: stuff, map_reduce, refine

    """
    uuid_srt = str(uuid.uuid4())
    async with aiofiles.open(f"{OUT_FILE_DIR}/{uuid_srt}.pdf", 'wb+') as out_file:
        content = await file.read()
        await out_file.write(content)

    res = summery(pdf_location=f"{OUT_FILE_DIR}/{uuid_srt}.pdf",
                  chain_type=Chain_Type.map_reduce)
    os.remove(f"{OUT_FILE_DIR}/{uuid_srt}.pdf")
    return res


@router.post("/text")
async def summaries_text(message: Summery_Text):
    """

    summarizes the text with provided chain type
    chain: stuff, map_reduce, refine

    """
    res = summery_from_text(text=message.text,
                            chain_type=Chain_Type.map_reduce)
    return res


@router.post("/url")
async def summaries_url(message: Summery_URL):
    try:
        loader = UnstructuredURLLoader(urls=[message.url])
        data = loader.load()
        res = summery_from_text(text=' '.join([
            d.page_content for d in data
        ]),
            chain_type=Chain_Type.map_reduce)
        return res
    except Exception as e:
        return ''
