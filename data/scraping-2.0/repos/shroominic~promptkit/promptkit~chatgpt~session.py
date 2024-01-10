from typing import List, TYPE_CHECKING
from pydantic import BaseModel
from dotenv import load_dotenv
from os import getenv
import openai
from promptkit.config import settings
from promptkit.chatgpt.schema import (
    BaseMessage,
    UserMessage,
    SystemMessage,
)

if TYPE_CHECKING:
    from promptkit.chatgpt.response import AssistantResponse


load_dotenv()
openai.api_key = getenv("OPENAI_API_KEY")


class ChatSession(BaseModel):
    """ Session with the chatgpt api """
    model: str = "gpt-3.5-turbo"
    history: List[BaseMessage] = []
    cost: float = 0
    
    def add_user(self, message: str) -> None:
        self.history.append(UserMessage(content=message))
    
    def add_system(self, message: str) -> None:
        self.history.append(SystemMessage(content=message))
    
    async def direct_response(self, instruction: str) -> "AssistantResponse":
        """ Add a system message as instruction and generate a response """
        self.add_system(instruction)
        return await self.get_response()
    
    async def get_response(self) -> "AssistantResponse":
        """ Get response and add it to the session history """
        response = await self.generate()
        self.history.append(response)
        return response
    
    async def generate(self, **kwargs) -> "AssistantResponse":
        """ Generate a response using the chatgpt api """
        history = [m.model_dump() for m in self.history]
        completion = await openai.ChatCompletion.acreate(
            api_key=settings.OPENAI_API_KEY,
            model=self.model,
            messages=history,
            **kwargs
        )
        
        # self.cost += completion.usage["total_tokens"] *  model_cost
        from promptkit.chatgpt.response import AssistantResponse  # workaround for circular import 
        return AssistantResponse(session=self, response=completion)  # type: ignore
    
    def __repr__(self) -> str:
        return f"ChatSession({self.model}, len: {len(self.history)})"
