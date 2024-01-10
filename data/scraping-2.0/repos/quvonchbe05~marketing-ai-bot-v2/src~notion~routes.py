from fastapi import APIRouter
from src.db.database import async_session_maker, vectorestore
from src.db.models import NotionModel
from src.notion.schemas import NotionSchema
from sqlalchemy import insert, select, update, delete, text
from pydantic import UUID4
from src.utils.validators import notion_validator
import json
import os
from typing import List
from src.ai.utils import ready_documetns

router = APIRouter(prefix="/api/notion", tags=["Notion integration"])


@router.get("/list")
async def notion_list():
    with open('src/uploads/notions.json', 'r') as f:
        data = json.load(f)
        
    return data

@router.post("/edit")
async def notion_create(
    notions: list[NotionSchema],
):
    for notion in notions:
        notion_validator(notion)

    with open("src/uploads/notions.json", "w") as f:
        json.dump(notions, f, indent=4)
        
    async with async_session_maker() as session:
        delete_query = text("DELETE FROM langchain_pg_embedding")
        await session.execute(delete_query)
        await session.commit()
            
    vectorestore.add_documents(ready_documetns)
        
    return {"message": "success"}


@router.delete("/delete/all")
async def notion_all_delete():
    with open("src/uploads/notions.json", "w") as f:
        json.dump([], f, indent=4)
        
    async with async_session_maker() as session:
        delete_query = text("DELETE FROM langchain_pg_embedding")
        await session.execute(delete_query)
        await session.commit()
            
    vectorestore.add_documents(ready_documetns)
    return {"message": "success"}
