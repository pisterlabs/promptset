import os
import openai
import uvicorn
import asyncio
import datetime
from loguru import logger
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from message_bodys import *
from database_option import models, database, crud, schemas
from util import get_key_usage, get_proxies, num_tokens_from_messages

# 加载环境变量
load_dotenv()
assert (api_key := os.getenv("OPENAI_API_KEY")), "Please configure OPENAI_API_KEY in the environment."

# 加载数据库
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins)


@app.post('/chat-process')
async def chat_process(data: ChatProcessRequest, db: Session = Depends(crud.get_db)):
    logger.info(data.json(ensure_ascii=False))
    if not data.prompt:
        return "Message can't be empty."
    chat_response = ChatProcessResponse()
    # 配置对话内容
    messages = []
    if data.options.parentMessageId:
        # 查询历史会话
        db_conversation = crud.get_conversation_by_id(db, id=data.options.parentMessageId)
        if db_conversation:
            messages = db_conversation.contents
    if not messages:
        messages.append({"role": "system", "content": data.systemMessage})
    # 插入最新问题
    messages.append({"role": "user", "content": data.prompt})
    model = _model if (_model := os.getenv("OPENAI_API_MODEL")) else "gpt-3.5-turbo-0301"
    # 计算token
    chat_response.question_token = await num_tokens_from_messages(messages, model=model)
    # openai请求参数
    params = {
        "model": model,
        "messages": messages,
        'temperature': data.temperature,
        'top_p': data.top_p,
        "stream": True
    }
    # 设置代理
    openai.proxy = get_proxies(package="openai")
    # 创建会话
    chat_reply_process = await asyncio.get_running_loop().run_in_executor(None, lambda: openai.ChatCompletion.create(**params))

    async def generate():
        for index, chat in enumerate(chat_reply_process):
            detail = chat.to_dict_recursive()
            choice = detail.get("choices")[0]
            delta = choice.get("delta")
            if not chat_response.role:
                chat_response.role = delta.get("role", "")
            if not chat_response.id:
                chat_response.id = detail.get("id", "")
            chat_response.text += delta.get("content", "")
            chat_response.delta = delta
            chat_response.detail = detail
            chat_response.answer_token = await num_tokens_from_messages([{"role": "system", "content":chat_response.text}], model=model)
            response = chat_response.json(ensure_ascii=False)
            yield f"\n{response}" if index else response

        consume_token = [chat_response.question_token, chat_response.answer_token]
        messages.append({"role": chat_response.role, "content": chat_response.text})
        # 更新会话消息
        if data.options.parentMessageId:
            # 若存在则只更新
            try:
                result = crud.update_conversation(db, id=data.options.parentMessageId, messages=messages, new_id=chat_response.id, consume_token=consume_token)
                logger.info(f"id:{result.id} title:{result.title} consume_token:{result.consume_token} messages:{result.contents}")
            except Exception as e:
                logger.error(f"Update error: id={data.options.parentMessageId} new_id={chat_response.id} consume_token:{consume_token} messages={messages}")
                logger.error(f"Insert error reason: {e.__str__()}")
        else:
            conversation = schemas.ConversationInsert(
                id=chat_response.id,
                user_id=None,
                title=data.prompt,
                contents=messages,
                create_time=datetime.datetime.now(),
                consume_token = consume_token
            )
            # 插入新的会话
            try:
                result = crud.insert_conversation(db, conversation)
                logger.info(f"id:{result.id} title:{result.title} consume_token:{result.consume_token} messages:{result.contents}")
            except Exception as e:
                logger.error(f"Insert error: {conversation.json(ensure_ascii=False)}")
                logger.error(f"Insert error reason: {e.__str__()}")

    # 流式返回
    return StreamingResponse(content=generate(), media_type="application/octet-stream")


@app.post("/config", response_model=ConfigResponse, summary="配置文件记录")
async def config():
    usage = await get_key_usage()
    response = ConfigResponse()
    response.data.balance = f"${usage}"
    logger.info(f"config: {response.json(ensure_ascii=False)}")
    return response


@app.post("/session", response_model=SessionResponse, summary="")
async def session():
    response = SessionResponse()
    response.data.auth = True if os.getenv("AUTH_SECRET_KEY") else False
    logger.info(f"session: {response.json(ensure_ascii=False)}")
    return response


@app.post("/verify", response_model=VerifyResponse, summary="验证授权")
async def verify(data: VerifyRequest):
    response = VerifyResponse()
    if data.token == os.getenv("AUTH_SECRET_KEY"):
        response.status = "Success"
        response.message = ""
        response.token = os.getenv("AUTH_SECRET_KEY")
    logger.info(f"verify: {response.json(ensure_ascii=False)}")
    return response


if __name__ == "__main__":
    uvicorn.run(app=f"{Path(__file__).stem}:app", host="0.0.0.0", port=3002, reload=True)
