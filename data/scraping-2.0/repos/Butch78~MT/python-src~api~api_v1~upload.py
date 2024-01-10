from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Depends
from sqlmodel import Session


import crud
from utils.deps import get_session
from services.index import LlamaIndex

router = APIRouter()


@router.post(
    "/{user_id}",
    response_description="Upload a file",
    tags=["upload"],
)
async def file_upload(
    user_id: str,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    file: UploadFile = File(...),
):
    if file.filename:
        print("User ID:", user_id)
        print("File Name:", file.filename)
        llama_index = LlamaIndex(user_id=user_id, file_name=file.filename)
        background_tasks.add_task(llama_index.upload_file, session, file)

    return {"message": "File uploaded successfully"}
