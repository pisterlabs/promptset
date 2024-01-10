import promptlayer
import os
from dotenv import load_dotenv
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class chatgpt():
    def __init__(self, model_name='gpt-3.5-turbo', pl_tags=["curio"]):
        load_dotenv()
        promptlayer.api_key = os.getenv("PROMPTLAYER_API_KEY")
        openai = promptlayer.openai
        # openai.organization = os.getenv("OPENAI_ORGANIZATION")
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.model_name = model_name
        self.pl_tags = pl_tags
        
    def chat_with_messages(self, messages):
        chat = PromptLayerChatOpenAI(pl_tags=self.pl_tags, model_name=self.model_name)
        response = chat(messages)
        return response
    
    def __call__(self, *args, **kwds):
        return self.chat_with_messages(*args, **kwds)
