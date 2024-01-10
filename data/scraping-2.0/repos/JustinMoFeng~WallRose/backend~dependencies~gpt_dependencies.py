
import os
import time

from openai import OpenAI, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletion

from dotenv import load_dotenv
load_dotenv()

from models.chat_models import Message, SystemMessage

system_message = SystemMessage(
    # 语言助手 System Prompt
    content = 
'''You are a "GPT" – a version of ChatGPT that has been customized for a specific use case. GPTs use custom instructions, capabilities, and data to optimize ChatGPT for a more narrow set of tasks. You yourself are a GPT created by a user, and your name is Makise Kurisu. Note: GPT is also a technical term in AI, but in most cases if the users asks you about GPTs assume they are referring to the above definition. Here are instructions from the user outlining your goals and how you should respond:
请扮演命运石之门中 Amadeus 里的牧濑红莉栖，拥有手机语音助手的功能，默认使用简体中文回答用户的问题。''',
)

client = AsyncOpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("BASE_URL"),
)

frontend_available_functions = ["set_alarm", "get_current_weather"]

tools = [
    {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Set an alarm for a specific time and date",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "The time to set the alarm for, in 24-hour format (HH:MM)"
                    },
                    "date": {
                        "type": "string",
                        "description": "The date to set the alarm for, in YYYY-MM-DD format"
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the alarm"
                    }
                },
                "required": ["time", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a specific location and date",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名，如北京，也可以为 'current' 表示当前位置"
                    },
                    "date": {
                        "type": "string",
                        "description": "The date to get the weather for, in YYYY-MM-DD format"
                    },
                },
                "required": ["location", "date"]
            },
        },
    },
]

from copy import deepcopy

async def gpt_api_stream(messages: list[Message]) -> AsyncStream[ChatCompletionChunk]:
    """
    为提供的对话消息创建新的回答 (流式传输)
    """
    try:
        # 拷贝一份消息
        messages = deepcopy(messages)
        messages[0].content += f"\nCurrent DateTime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
        print(messages)
        stream : AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[message.model_dump() for message in messages],
            tools=tools,
            stream=True,
        )
        if not isinstance(stream, AsyncStream):
            raise TypeError("Expected AsyncStream")
        return stream
    except Exception as err:
        print(f"OpenAI API 异常: {err}")
        raise err