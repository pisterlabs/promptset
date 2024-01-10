from typing import Any
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from utils import messages_to_prompt


class ChatGPT:
    def __init__(self, config):
        self.config = config
        self.model = ChatOpenAI(
            model_name=self.config["model_name"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.model(*args, **kwds)

    def get_num_tokens(self, text):
        return self.model.get_num_tokens(text)

    async def agenerate(self, messages):
        resp = await self.model._agenerate(messages=messages)
        return resp.generations[0].text


type_to_class = {
    "chatgpt": ChatGPT,
}


def load_model(config):
    model_type = config["type"]
    return type_to_class[model_type](config)
