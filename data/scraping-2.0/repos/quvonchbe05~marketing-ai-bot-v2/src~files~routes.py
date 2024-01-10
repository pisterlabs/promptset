from fastapi import APIRouter, UploadFile, HTTPException
from src.db.database import async_session_maker, vectorestore
from src.db.models import FilesModel
from sqlalchemy import insert, select, delete, text
import os
from typing import List
from pydantic import UUID4
from src.utils.validators import file_upload_validator
from src.ai.utils import ready_documetns

router = APIRouter(prefix="/api/files", tags=["File system"])


@router.post("/file/upload")
async def document_upload(files: List[UploadFile]):
    file_paths = []
    validated_files = file_upload_validator(files)
    async with async_session_maker() as session:
        for file in validated_files:
            file_path = f"src/uploads/{file.filename}"

            with open(file_path, "wb") as f:
                f.write(file.file.read())
                
            stmt = select(FilesModel).where(FilesModel.file_path == file_path)
            document = await session.scalar(stmt)
            
            if not document:
                stmt = insert(FilesModel).values(file_path=file_path, name=file.filename)
                await session.execute(stmt)

            file_paths.append(file_path)
            
        delete_query = text("DELETE FROM langchain_pg_embedding")
        await session.execute(delete_query)
        await session.commit()
            
    vectorestore.add_documents(ready_documetns)

    return {"files": file_paths}


@router.get("/list")
async def document_list():
    async with async_session_maker() as session:
        stmt = select(FilesModel)
        response = await session.scalars(stmt)
    return response.all()


@router.delete("/delete/{id}")
async def document_delete(
        id: UUID4,
):
    try:
        async with async_session_maker() as session:
            stmt = select(FilesModel).where(FilesModel.id == id)
            document = await session.scalar(stmt)

            os.remove(document.file_path)

            delete_stmt = delete(FilesModel).where(FilesModel.id == id)
            await session.execute(delete_stmt)
            
            delete_query = text("DELETE FROM langchain_pg_embedding")
            await session.execute(delete_query)
            await session.commit()
            
        vectorestore.add_documents(ready_documetns)

        return {"message": "success"}
    except:
        raise HTTPException(status_code=404, detail="File not found!")


@router.delete("/delete")
async def document_all_delete():
    async with async_session_maker() as session:
        stmt = select(FilesModel)
        documents = await session.scalars(stmt)
        for document in documents:
            try:
                os.remove(document.file_path)
            except:
                pass

        delete_stmt = delete(FilesModel)
        await session.execute(delete_stmt)
        
        delete_query = text("DELETE FROM langchain_pg_embedding")
        await session.execute(delete_query)
        await session.commit()

    return {"message": "success"}


