import openai
from openai import AsyncOpenAI
import os
import asyncio
from models.user_models import UserInDB
from dependencies.auth_dependencies import get_current_user

from dotenv import load_dotenv
load_dotenv()

from sse_starlette.sse import EventSourceResponse

from fastapi import APIRouter, Depends

router = APIRouter()

async def gpt_api_stream(prompt):
    """
    为提供的对话消息创建新的回答 (流式传输)
    """
    try:
        client = AsyncOpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("BASE_URL")
        )
        stream = await client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
        )
        async for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
    except Exception as err:
        print(f"OpenAI API 异常: {err}")    

async def produce_greeting_stream():
    """
    每秒钟返回 Greeting 中的一个字母
    """
    for char in "你好，我是牧濑红莉栖，有什么问题吗?":
        yield {"event": "greeting", "data": char}
        await asyncio.sleep(0.2)
    
@router.get("/greeting_stream")
async def greeting_stream():
    """
    以 SSE 流的形式返回 Greeting
    """
    return EventSourceResponse(produce_greeting_stream())

if __name__ == "__main__":
    prompt = "There are 9 birds in the tree, the hunter shoots one, how many birds are left in the tree？"
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(gpt_api_stream(prompt))
    print("Done")
