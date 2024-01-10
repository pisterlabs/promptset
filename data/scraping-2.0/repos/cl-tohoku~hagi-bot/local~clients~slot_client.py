from .models.openai_model import OpenAIChat
from .models.message import Message
from .models.config import ModelConfig


class SlotClient(OpenAIChat):

    def __init__(self, *, model_config: ModelConfig, input_history_length: int) -> None:
        super().__init__(model_config=model_config)
        self.load_system_instruction()
        self.input_histrory_length = input_history_length
    
    def update_main_history(self, chat_history: list[Message]) -> None:
        if self.input_histrory_length is None:
            self.input_chat_history = chat_history
        else:
            self.input_chat_history = chat_history[-self.input_histrory_length:]

    @property
    def chat_history_text(self) -> str:
        texts = list()
        for message in self.input_chat_history:
            texts.append(f"{message.role.main_character_name}: {message.content}")
        return "\n".join(texts)
