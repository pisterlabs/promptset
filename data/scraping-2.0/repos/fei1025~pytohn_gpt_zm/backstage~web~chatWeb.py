import os

from fastapi import APIRouter, Query, UploadFile, File, Form

from fastapi import Depends, Request
from sqlalchemy.orm import Session

import openAi.openAIChat as openAichat
import openAi.openAiUtil as openAiUtil
import openAi.KnowledgeChat as knowledgeChat
from Util.result import Result
from db.database import engine, get_db
from entity import models, schemas, crud

from sse_starlette.sse import EventSourceResponse

from entity.schemas import reqChat, userSetting
import fun.Knowledge as kn

router = APIRouter()


@router.get("/users")
async def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]


@router.get("/getUserSetting")
async def get_user_setting(db: Session = Depends(get_db)):
    return Result.success(data=crud.get_user_setting(db))


@router.get("/get_all_model")
async def get_all_model():
    return Result.success(data=openAiUtil.get_all_model())


@router.post("/saveUserSetting")
async def save_user_setting(setting: userSetting, db: Session = Depends(get_db)):
    userSetting = models.User_settings()
    userSetting.model = setting.model
    userSetting.http_proxy = setting.httpProxy
    userSetting.wolfram_appid = setting.wolframAppid
    userSetting.openai_api_base = setting.openaiApiBase
    userSetting.openai_api_key = setting.openaiApiKey
    userSetting.theme = setting.theme
    crud.save_user_setting(db, userSetting)
    return Result.success()


@router.get("/delete_chat")
async def delete_chat(chatId: str, db: Session = Depends(get_db)):
    crud.delete_chat(db, chatId)
    return Result.success()


@router.get("/delete_all_chat")
async def delete_all_chat(db: Session = Depends(get_db)):
    crud.delete_all_chat(db)
    return Result.success()


@router.get("/getAllHist")
async def get_all_Hist(type:str,db: Session = Depends(get_db)):
    list = crud.get_all_Hist(db,type)
    return Result.success(list)


@router.get("/getChatHistDetails")
async def get_chat_hist_details(db: Session = Depends(get_db), chatId: str = Query(...)):
    return Result.success(crud.get_chat_hist_details(db, chatId))


@router.post("/save_chat_hist")
async def save_chat_hist(res: reqChat, db: Session = Depends(get_db)):
    chatHist = models.chat_hist()
    chatHist.title = res.content
    crud.save_chat_hist(db, chatHist)
    return Result.success([chatHist])


@router.post("/update_chat")
async def update_chat(res: reqChat, db: Session = Depends(get_db)):
    print(res)
    chatHist = models.chat_hist()
    chatHist.chat_id = res.chat_id
    chatHist.model = res.model
    chatHist.title = res.title
    crud.update_chat(db, chatHist)
    return Result.success()


# @router.post("/send_open_ai")
# @router.post("/update_chat")
async def send_open_ai1(request: Request):
    # 获取请求中的所有数据
    all_data = await  request.json()
    print(f"接受到的所有数据:{all_data}")
    # 获取reqChat对象中的特定字段
    # chat_id = res.chat_id
    # content = res.content
    # role = res.role


@router.post("/send_open_ai")
def send_open_ai(request: Request, res: reqChat, db: Session = Depends(get_db)):
    if res.chat_id:
        print("没有保存")
        pass
    else:
        print("保存历史记录")
        chatHist = models.chat_hist()
        chatHist.title = res.content
        chatHist.type = '0'
        crud.save_chat_hist(db, chatHist)
        res.chat_id = chatHist.chat_id

    # 保存历史记录
    chatHistDetails = models.chat_hist_details()
    chatHistDetails.chat_id = res.chat_id
    chatHistDetails.content = res.content
    chatHistDetails.role = res.role
    crud.save_chat_hist_details(db, chatHistDetails)

    async def event_generator():
        result = openAichat.send_open_ai(db, res)
        print(result)
        content = ""
        # assistant
        for i in result:
            if await request.is_disconnected():
                print("连接已中断")
                break
            if "stop" != i.choices[0].finish_reason:
                content = content + i.choices[0].delta.content
                yield i.choices[0].delta.content
        chatHistDetails = models.chat_hist_details()
        chatHistDetails.chat_id = res.chat_id
        chatHistDetails.content = content
        chatHistDetails.role = "assistant"
        crud.save_chat_hist_details(db, chatHistDetails)

    g = event_generator()
    return EventSourceResponse(g)


file_type = [
    ".pdf", ".docx", ".pptx", ".epub", ".xlsx"
]
