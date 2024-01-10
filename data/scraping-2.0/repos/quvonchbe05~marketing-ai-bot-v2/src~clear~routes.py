from fastapi import APIRouter
from src.db.database import async_session_maker, vectorestore
from src.ai.utils import ready_documetns
from src.db.models import FilesModel, NotionModel, ChatHistoryModel
from sqlalchemy import select, delete, text
import os


router = APIRouter(prefix="/api", tags=["Clear Knowledge Base"])


@router.delete("/clear")
async def clear_knowladge_base():
    async with async_session_maker() as session:
        document_stmt = select(FilesModel)
        documents = await session.scalars(document_stmt)
        for document in documents:
            try:
                os.remove(document.file_path)
            except:
                pass

        delete_stmt = delete(FilesModel)
        await session.execute(delete_stmt)

        delete_stmt = delete(ChatHistoryModel)
        await session.execute(delete_stmt)

        notion_delete_stmt = delete(NotionModel)
        await session.execute(notion_delete_stmt)
        
        delete_query = text("DELETE FROM langchain_pg_embedding")
        await session.execute(delete_query)
        await session.commit()

    return {"message": "success"}
