import os
import openai

from langchain.chat_models import ChatOpenAI

from .base import BaseModel

class ChatGPT(BaseModel):
    human_prefix = "human"
    ai_prefix = "ai"
    def __init__(self, retriever, conversations, carerInput, medicalInput, device) -> None:
        super().__init__(retriever, conversations, carerInput, medicalInput, device)

        api_key = os.getenv("OPENAI_API_KEY")
        temperature = os.getenv("OPENAI_TEMPERATURE", 0.7)
        openai.api_key = api_key    
        print(temperature)
        self.model = ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo")
