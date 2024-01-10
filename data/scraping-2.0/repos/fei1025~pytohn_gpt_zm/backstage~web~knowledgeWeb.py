from fastapi import APIRouter
import os

from fastapi import APIRouter, Query, UploadFile, File, Form

from fastapi import Depends, Request
from sqlalchemy.orm import Session

import openAi.openAIChat as openAichat
import openAi.KnowledgeChat as knowledgeChat
from Util.result import Result
from db.database import engine, get_db
from entity import models, schemas, crud

from sse_starlette.sse import EventSourceResponse

from entity.schemas import reqChat, userSetting
import fun.Knowledge as kn

router = APIRouter(
    prefix='/knowledge',
    tags=['知识库']
)


@router.post("/load_vectorstore")
def loadVectorstore(knowledgeId: int, db: Session = Depends(get_db)):
    knowledge = models.knowledge(id=knowledgeId)
    knowledgeChat.loadVectorstore(db, knowledge)
    return Result.success()


@router.post("/getAllKnowledge")
def getAllKnowledge(db: Session = Depends(get_db)):
    knowledgeList = crud.get_all_knowledge(db)
    return Result.success(knowledgeList)

@router.post("/getKnowledgeDetail")
def getKnowledgeDetail(knowledgeId:int,db: Session = Depends(get_db)):
    pass


@router.post("/send_open_ai")
def send_open_ai(request: Request, res: reqChat, db: Session = Depends(get_db)):
    if res.chat_id:
        print("没有保存")
        pass
    else:
        print("保存历史记录")
        chatHist = models.chat_hist()
        chatHist.title = res.content
        chatHist.type='1'
        crud.save_chat_hist(db, chatHist)
        res.chat_id = chatHist.chat_id

    # 保存历史记录
    chatHistDetails = models.chat_hist_details()
    chatHistDetails.chat_id = res.chat_id
    chatHistDetails.content = res.content
    chatHistDetails.role = res.role
    crud.save_chat_hist_details(db, chatHistDetails)

    async def event_generator():
        data_generator = knowledgeChat.send_open_ai(db, res)
        if await request.is_disconnected():
            print("连接已中断")
        for data_point in data_generator:
            yield data_point

    g = event_generator()
    return EventSourceResponse(g)


@router.post("/upload_check")
async def upload_check(request: Request, db: Session = Depends(get_db)):
    all_data = await  request.json()
    knowledge_file = crud.get_knowledge_file_ma5(db, all_data['md5'])
    if knowledge_file is None:
        return Result.success()
    else:
        return Result.error("文件已存在")


@router.post("/uploadKnowledge")
async def upload_file_Knowledge(file: UploadFile = File(...), fileId: str = Form(...), md5: str = Form(...),
                                db: Session = Depends(get_db)):
    # 获取文件内容
    file_content = await file.read()
    # 指定保存路径
    save_directory = "uploads"

    # 如果目录不存在，递归创建目录
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 拼接保存路径
    save_path = os.path.join(save_directory, file.filename)
    # 保存文件到指定路径
    with open(save_path, "wb") as f:
        f.write(file_content)
    know = models.knowledge(file_path=save_path)
    kn.create_knowledge(know, db)
    kow_file = models.knowledge_file(
        knowledge_id=know.id,
        content_md5=md5,
        file_name=file.filename,
        file_path=save_path
    )
    crud.save_knowledge_file(db, kow_file)
    return {"filename": file.filename, "fileId": fileId, "save_path": save_path}


