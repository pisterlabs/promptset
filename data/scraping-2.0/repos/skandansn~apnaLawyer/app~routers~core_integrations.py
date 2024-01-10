from fastapi import APIRouter, Depends, UploadFile, status, HTTPException

import os

from ..database import get_db
from sqlalchemy.orm import Session
from .. import models, schemas, oauth2
from ..apnaLawyer import langchain_query_processor, document_input_feeder, audio_transcribe

router = APIRouter()


@router.post('/query', response_model=schemas.QueryOutput)
async def query(input_query: schemas.QueryInput, db: Session = Depends(get_db),
                user_id: str = Depends(oauth2.require_user)):
    user = db.query(models.User).filter(models.User.id == user_id).first()

    processor_result = await langchain_query_processor(input_query, user)

    print(processor_result)
    return schemas.QueryOutput(**processor_result)


@router.post('/upload-files')
async def create_upload_file(files: list[UploadFile], db: Session = Depends(get_db),
                             user_id: str = Depends(oauth2.require_user)):
    user = db.query(models.User).filter(models.User.id == user_id).first()

    if user.role == 0:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail='This feature is only available for paid tier customers.')

    os.makedirs("./storage/files/" + user.email, exist_ok=True)

    for file in files:
        file_path = os.path.join("./storage/files/" + user.email, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

    await document_input_feeder(user.email)

    return {"Uploaded filenames": [file.filename for file in files]}


@router.get('/list-files')
async def list_user_files(db: Session = Depends(get_db), user_id: str = Depends(oauth2.require_user)):
    user = db.query(models.User).filter(models.User.id == user_id).first()

    os.makedirs("./storage/files/" + user.email, exist_ok=True)

    files = os.listdir("./storage/files/" + user.email)

    return {"Your uploaded files": [file for file in files]}


@router.post('/process-audio')
async def upload_audio(db: Session = Depends(get_db), user_id: str = Depends(oauth2.require_user)):
    user = db.query(models.User).filter(models.User.id == user_id).first()

    if user.role == 0:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail='This feature is only available for paid tier customers.')

    processor_result = await audio_transcribe(user)

    return schemas.QueryOutput(**processor_result)
