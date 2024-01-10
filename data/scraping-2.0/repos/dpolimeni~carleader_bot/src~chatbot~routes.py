from fastapi import APIRouter
import uuid
from fastapi.exceptions import HTTPException
from src.chatbot.schemas import Conversation, ChatMessage, Message
from src.schemas import OpenaiConfig
from src.config import configuration
from src.chatbot.service import QaService
from src.chatbot.utils import init_client_tools, execute_client_tools
from langchain.chat_models import ChatOpenAI
from openai import OpenAI

router = APIRouter()

chats = {}
chat_llm = ChatOpenAI(
    temperature=0,
    openai_api_key=configuration.openai_key,
    model=configuration.chat_model_version,
    request_timeout=15,
)
tools = init_client_tools()


@router.get("/chat", response_model=Conversation)
async def get_messages(chat_id: str):
    messages = chats.get(chat_id)
    try:
        return Conversation(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No chat found with id {chat_id}")


@router.post("/chat", response_model=ChatMessage)
async def chat(message: ChatMessage):
    user = message.chat_id if message.chat_id else str(uuid.uuid4())
    query = message.message

    chats[user] = [Message(sender="Cliente", message=query)]

    ## TODO save messages somewhere
    roles = {"AI": "assistant", "Cliente": "user"}
    conversation = [
        {"role": roles[m.sender], "content": m.message} for m in chats[user]
    ]

    openai_client = OpenAI(api_key=configuration.openai_key)
    response = openai_client.chat.completions.create(
        model=configuration.chat_model_version,
        messages=[
            {
                "role": "system",
                "content": """Il tuo compito è servire i clienti di un concessionario. 
Se il cliente non è specifico sul tipo di macchina che gli interessa chiedigli delle informazioni per proporgli quella più adatta.""",
            },
            *conversation,
        ],
        tools=tools,
    )
    response_message = response.choices[0].message
    if response_message.tool_calls:
        response = execute_client_tools(response_message, openai_client=openai_client)
        response_text = response["message"]
        response_extra = response["extra"]
    else:
        response_text = response_message.content
        response_extra = ""

    chats[user].extend([Message(sender="AI", message=response_text)])
    return ChatMessage(
        **{
            "sender": "AI",
            "message": response_text,
            "chat_id": user,
            "extra": response_extra,
        }
    )
