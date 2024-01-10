import json

from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse
from motor.motor_asyncio import AsyncIOMotorCollection
from bson.objectid import ObjectId

from fastapi import APIRouter, Depends, HTTPException

from dotenv import load_dotenv
load_dotenv()

from models.user_models import UserInDB
from models.chat_models import *
from database import get_chat_collection
from dependencies.auth_dependencies import get_current_user

router = APIRouter()

@router.get("/chats/list", response_model=list[Chat])
async def get_chats_list(
    page: int = 1,
    page_size: int = 10,
    user: UserInDB = Depends(get_current_user),
    collection : AsyncIOMotorCollection = Depends(get_chat_collection)
):
    """
    获取所有聊天列表
    """
    offset = (page-1)*page_size
    print(user.user_id)
    chats = await collection.find({"owner": str(user.user_id)}).sort("_id", -1).skip(offset).limit(page_size).to_list(None)
    return chats

@router.get("/chats/{chat_id}", response_model=Chat)
async def get_chat(
    chat_id: str,
    user: UserInDB = Depends(get_current_user),
    collection : AsyncIOMotorCollection = Depends(get_chat_collection)
):
    """
    获取聊天
    """

    res = await collection.find_one({"_id": ObjectId(chat_id)})
    
    if not res:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    chat = Chat(**res)

    if chat.owner != user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    return chat
    
@router.post("/chats/create", response_model=Chat)
async def create_chat(
    name: str = "新聊天",
    user: UserInDB = Depends(get_current_user),
    collection : AsyncIOMotorCollection = Depends(get_chat_collection)
):
    """
    创建聊天
    """
    chat = Chat(owner=user.user_id, name=name, messages=[])
    res = await collection.insert_one(chat.model_dump(exclude_none=True))
    chat.id = str(res.inserted_id)
    return chat

@router.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    user: UserInDB = Depends(get_current_user),
    collection : AsyncIOMotorCollection = Depends(get_chat_collection)
):
    """
    删除聊天
    """
    res = await collection.find_one({"_id": ObjectId(chat_id)})
    
    if not res:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    chat = Chat(**res)

    if chat.owner != user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    await collection.delete_one({"_id": ObjectId(chat_id)})
    return {"message":"Delete Success"}
    
from dependencies.gpt_dependencies import system_message, gpt_api_stream
    
@router.post("/chats/{chat_id}/send", response_class=EventSourceResponse)
async def send_message(
    chat_id: str,
    content: str | None = None,
    user: UserInDB = Depends(get_current_user),
    collection : AsyncIOMotorCollection = Depends(get_chat_collection)
):
    """
    发送消息，存入数据库，并以流式响应返回
    """
    res = await collection.find_one({"_id": ObjectId(chat_id)})
    chat = Chat(**res)

    # 如果用户不是聊天室的拥有者，返回错误
    if chat.owner != user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    # 如果messages为空用系统消息初始化
    if not chat.messages:
        chat.messages = [system_message]

    # 如果最后一条消息是 assistant 的工具调用，返回错误
    if chat.messages[-1].role == "assistant" and chat.messages[-1].tool_calls:
        raise HTTPException(status_code=403, detail="Need to response to tool call")

    # 若 content 不为空，是用户发送的消息，为空说明是返回工具调用后继续请求 GPT 的回答
    if content:
        chat.messages.append(UserMessage(content=content))
        collection.update_one({"_id": ObjectId(chat_id)},{"$set":chat.model_dump(exclude={"id"})})

    stream = await gpt_api_stream(chat.messages)

    # 以流式响应返回
    async def produce_gpt_stream():
        """
        以 SSE 流的形式返回 GPT 的回答
        """
        assistant_message = AssistantMessage(content="")
        tool_calls = []

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)
                assistant_message.content += delta.content
                yield {"event": "message", "data": delta.content}
            elif delta and delta.tool_calls:
                for tool_call in delta.tool_calls:
                    index = tool_call.index
                    function = tool_call.function
                    if tool_call.id:
                        tool_calls.append({})
                        tool_calls[index]["id"] = tool_call.id
                        tool_calls[index]["type"] = tool_call.type

                    if function.name:
                        print(f"Start calling {function.name}")
                        yield {"event": "start_call", "data": function.name}
                        tool_calls[index]["function"] = {
                            "name": function.name,
                            "arguments": ''
                        }
                    if function.arguments:
                        tool_calls[index]["function"]["arguments"] += function.arguments
            elif chunk.choices[0].finish_reason:
                print(f"\nFinished: {chunk.choices[0].finish_reason}")
                yield {"event": "finish", "data": chunk.choices[0].finish_reason}

        if tool_calls:
            print("Tool calls:")
            print(tool_calls)
        
        # call_message = AssistantMessage(tool_calls=[ToolCall(**tool_call) for tool_call in tool_calls] if tool_calls else None)
        if tool_calls:
            assistant_message.tool_calls = [ToolCall(**tool_call) for tool_call in tool_calls]

        chat.messages.append(assistant_message)

        # for tool_call in tool_calls:
        #     print(f"Fake Calling {tool_call}")
        #     chat.messages.append(
        #         ToolMessage(
        #             tool_call_id=tool_call["id"],
        #             name=tool_call["function"]["name"],
        #             content="Success",
        #         )
        #     )

        collection.update_one({"_id": ObjectId(chat_id)},{"$set": chat.model_dump(exclude={"id"})})

        for tool_call in tool_calls:
            yield {"event": "tool_call", "data": json.dumps(tool_call)}

    return EventSourceResponse(produce_gpt_stream())

@router.post("/chats/{chat_id}/return")
async def call_tool_return(
    chat_id: str,
    tool_name: str,
    tool_call_id: str,
    content: str,
    user: UserInDB = Depends(get_current_user),
    collection : AsyncIOMotorCollection = Depends(get_chat_collection)
):
    """
    调用工具
    """
    res = await collection.find_one({"_id": ObjectId(chat_id)})
    chat = Chat(**res)

    # 如果用户不是聊天室的拥有者，返回错误
    if chat.owner != user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # 将消息存入数据库
    chat.messages.append(ToolMessage(tool_call_id=tool_call_id, name=tool_name, content=content))
    collection.update_one({"_id": ObjectId(chat_id)},{"$set":chat.model_dump(exclude={"id"})})

    return {"message":"Return Success"}