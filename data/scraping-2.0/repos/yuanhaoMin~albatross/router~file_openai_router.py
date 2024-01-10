from fastapi import APIRouter, Depends, File, Form, HTTPException
from schema.openai_completion_schema import UpdateCompletionResponse
from service import openai_file_service
from sqlalchemy.orm import Session
from typing import Annotated
from util.db_util import get_db

router = APIRouter(
    prefix="/file/openai",
    tags=["file openai"],
    responses={404: {"description": "Not found"}},
)


@router.post("/pdf")
def create_upload_pdf(
    pdf_file: Annotated[bytes, File()],
    username: Annotated[str, Form()],
    template_id: Annotated[int, Form()],
    db: Session = Depends(get_db),
) -> UpdateCompletionResponse:
    if not pdf_file:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot upload empty PDF file",
        )
    return openai_file_service.upload_pdf_in_completion(
        pdf_file, username, template_id, db
    )
