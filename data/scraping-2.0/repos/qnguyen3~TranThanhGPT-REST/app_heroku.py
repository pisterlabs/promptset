from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import json
from langchain.agents import initialize_agent
from dotenv import load_dotenv
from tools import thanh_singer, thanh_life_lesson, thanh_kol, thanh_lawsuit, thanh_private, is_json
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentType
from prompts import (
    PREFIX,
    SUFFIX,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatInput(BaseModel):
    message: str

class ChatOutput(BaseModel):
    type: str
    content: Any

@app.post("/chat", response_model=ChatOutput)
async def chat(chat_input: ChatInput) -> ChatOutput:

    question = chat_input.message

    tools = [
    Tool(
        name = "Thành Dạy Hát",
        func=thanh_singer.run,
        description="Có ích khi có câu hỏi muốn Trấn Thành hát hoặc muốn Trấn Thành dạy cho cách hát",
        return_direct=True
    ),
    Tool(
        name = "Thành Đạo Lý",
        func=thanh_life_lesson.run,
        description="Có ích khi người dùng hỏi Trấn Thành các câu hỏi liên quan đến kinh nghiệm sống và tình yêu",
        return_direct=True
    ),
    Tool(
        name = "Thành Nổi Tiếng",
        func=thanh_kol.run,
        description="Có ích khi có câu hỏi liên quan đến nghề nghiệp và sự nổi tiếng của Trấn Thành, input nên là câu hỏi trực tiếp của người dùng",
        return_direct=True
    ),
    Tool(
        name = "Thành Kiện Cáo",
        func=thanh_lawsuit.run,
        description="Có ích khi người dùng nói những câu nói sẽ gây ảnh hưởng xấu đến hình ảnh của Trấn Thành (bao gồm việc sử dụng các chất kích thích, đua xe, hoặc bất kì hành đồng nào có thể gây ảnh hưởng đến hình ảnh của Trấn Thành). Input phải chính là USER'S INPUT và KHÔNG THAY ĐỔI bất kì điều gì",
        return_direct=True
    ),
    Tool(
        name = "Thành Riêng Tư",
        func=thanh_private.run,
        description="Có ích khi người dùng hỏi hoặc nói những câu hỏi liên quan đến sự riêng tư của Trấn Thành",
        return_direct=True
    ),]
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
    agent_kwargs = {'human_message': SUFFIX, 'system_message': PREFIX}
    agent_chain = initialize_agent(tools, ChatOpenAI(temperature=0), agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=False, agent_kwargs=agent_kwargs, memory=memory)

    out = agent_chain.run(question)

    if is_json(out):
        json_obj = json.loads(out)
        type = json_obj['type']
        content = json_obj['content']
        return {'type': type, 'content': content}
    else:
        return {'type': 'text', 'content': out}
