from fastapi import FastAPI
from langchain.schema import messages_to_dict
from pydantic import BaseModel
from typing import List
from main import Chatbot

chatbot = Chatbot(use_local_history=False)

app = FastAPI()

class UserInput(BaseModel):
    message: str


class BotResponse(BaseModel):
    answer: str


class ChatHistory(BaseModel):
    messages: List[str]


@app.post("/chat")
def chat_with_bot(user_input: UserInput):
    input_message = user_input.message

    chatbot.handle_input(input_message)

    history_messages = chatbot.memory.chat_memory.messages

    response_data = {
        "answer": chatbot.conversation_chain(input_message)['answer'],
        "chat_history": messages_to_dict(history_messages)
    }

    return response_data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
