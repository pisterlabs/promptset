from dto import ChatbotRequest
from samples import list_card
import aiohttp
import time
import logging
import openai

import os
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)

def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

# 환경 변수 처리 필요!
openai.api_key = ''
SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."
logger = logging.getLogger("Callback")

async def callback_handler(request: ChatbotRequest) -> dict:
    os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

    raw_data = read_file("data/project_data_kakaosync.txt")

    system_message = "assistant는 챗봇으로 동작한다. 챗봇은 '제품정보' 내용을 참고하여, user의 질문 혹은 요청에 따라 적절한 답변을 제공합니다."
    human_template = ("제품정보: {product_data}\n" +
                      request.userRequest.utterance )

    system_message_prompt = SystemMessage(content=system_message)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat = ChatOpenAI(temperature=0.8)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    output_text = chain.run(product_data=raw_data)

    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }

    # debug
    print(output_text)

    time.sleep(1.0)

    url = request.userRequest.callbackUrl

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload) as resp:
                await resp.json()


async def callback_handler2(request: ChatbotRequest) -> dict:

    # ===================== start =================================
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": request.userRequest.utterance},
        ],
        temperature=0,
    )
    # focus
    output_text = response.choices[0].message.content

   # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    time.sleep(1.0)

    url = request.userRequest.callbackUrl

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload, ssl=False) as resp:
                await resp.json()