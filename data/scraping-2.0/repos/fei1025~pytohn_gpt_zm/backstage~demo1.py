import asyncio
from http.client import HTTPException
from typing import List

import uvicorn
from fastapi import FastAPI, Depends, Request
from sqlalchemy.orm import Session

import openAi.langChat
from db.database import engine, get_db
from entity import models, schemas, crud

from sse_starlette.sse import EventSourceResponse

from web import chatWeb,knowledgeWeb


models.Base.metadata.create_all(bind=engine)

app = FastAPI()
# 计算引擎
# https://python.langchain.com/docs/integrations/providers/wolfram_alpha


app.include_router(chatWeb.router)
app.include_router(knowledgeWeb.router)


@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.post("/users/{user_id}/items/", response_model=schemas.Item)
def create_item_for_user(
        user_id: int, item: schemas.ItemCreate, db: Session = Depends(get_db)
):
    return crud.create_user_item(db=db, item=item, user_id=user_id)


@app.get("/items/", response_model=List[schemas.Item])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = crud.get_items(db, skip=skip, limit=limit)
    return items


@app.get("/get_user_setting/")
def get_user_setting(db: Session = Depends(get_db)):
    return crud.get_user_setting(db)


@app.get("/getSee")
async def root(request: Request):
    async def event_generator(request: Request):
        res_str = "七夕情人节即将来临，我们为您准备了精美的鲜花和美味的蛋糕"
        for i in res_str:
            if await request.is_disconnected():
                print("连接已中断")
                break
            yield {
                "event": "message",
                "retry": 15000,
                "data": i
            }

            await asyncio.sleep(0.1)

    g = event_generator(request)
    return EventSourceResponse(g)


@app.post("/chat")
async def chat(request: Request, user_id: int):
    print(user_id)

    async def event_generator(request: Request):
        result = openAi.langChat.get_chat()
        for i in result:
            if await request.is_disconnected():
                print("连接已中断")
                break
            yield i.message.content

    g = event_generator(request)
    return EventSourceResponse(g)


if __name__ == '__main__':
    uvicorn.run(app, port=6688, host="0.0.0.0")
