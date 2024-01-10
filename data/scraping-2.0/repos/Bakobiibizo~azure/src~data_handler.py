import base64
from typing import List, Optional
from src.generation.openai_text import OpenAITextGeneration
from src.messages.context import ContextWindow
from src.messages.create_messages import CreateMessage, Message
from openai.openai_object import OpenAIObject


class DataHandler:
    def __init__(self, persona_image=None):
        self.openai_text: OpenAITextGeneration = OpenAITextGeneration()
        self.context_window: ContextWindow = ContextWindow()
        self.image_path: Optional[str] = persona_image
        self.context: List[Message] = self.context_window.context
        self.create_messages: CreateMessage = CreateMessage()

    def handle_chat(self, role, content, context_window=None, primer_choice=None) -> OpenAIObject:
        if not self.context:
            self.context = self.context_window.create_context(
                role=role, content=content, context_window=context_window, primer_choice=primer_choice
                )
        else:
            message: Message = self.create_messages.create_message(
                role=role, content=content
            )
            self.context = self.context_window.add_message(message)
        response = self.openai_text.send_chat_complete(self.context)
        return response
    
    def handle_ai_chat(self, message: Message) -> None:
        self.context = self.context_window.add_message(message)

    def handle_image(self) -> str:
        with open(self.image_path, "rb") as f:
            content = f.read()
        return base64.b64encode(content).decode("utf-8")
