# services\ask_question_service.py
import openai
from typing import Dict
# from logger import logger
from aiohttp import ClientSession
from .utils import handle_exception, logger_stream_handler

openai.aiosession.set(ClientSession())

async def ask_question(input_text: str, envs: Dict[str, str]) -> str:
    try:
        openai.api_key = envs["OPENAI_API_KEY"]
        messages = [{"role": "user", "content": f"Answer the following question as best you can: {input_text}" }]
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )

        # logger.debug("ask_question: %s", response)
        logger_stream_handler(f"logger_stream_handler test: {response}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        handle_exception(e, "ask_question")
