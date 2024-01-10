from datetime import date
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage
)
from slack_app.util.routine import get_logger


DEFAULT_MODEL_NAME = 'gpt-4-1106-preview'
KNOWLEDGE_CUTOFF  = '2023-04'
SYSTEM_CONTENT = f'You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. \
Knowledge cutoff: {KNOWLEDGE_CUTOFF} Current date: {date.today()}'
BOT_USER_ID = '@U0504NZFL4S'

logger = get_logger(__name__)


class ChatGPT:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.chat = ChatOpenAI(model_name=model_name, request_timeout=180)

    def chat_request(self, messages: list[dict[str: str]]) -> str:
        response = self.chat(messages=messages)
        return response.content

    def run(self, messages: list[dict[str, str]]) -> str:
        chatgpt_messages = self._prepare_messages(messages)
        logger.info(f'chatgpt request: {chatgpt_messages}')
        res = self.chat_request(chatgpt_messages)
        logger.info(f'chatgpt response: {res}')
        return res
    
    def _remove_mention(self, content: str) -> str:
        return content.replace(f'<{BOT_USER_ID}>', '').strip()
    
    def _prepare_messages(self, messages: list[dict[str, str]]) -> list[BaseMessage]:
        chatgpt_messages = [
            SystemMessage(content=SYSTEM_CONTENT)
        ]
        for message in messages:
            content = self._remove_mention(message['content'])
            if message['role'] == 'bot' or message['role'] == BOT_USER_ID[1:]:
                _message = AIMessage(content=content)
            else:
                _message = HumanMessage(content=content)
            chatgpt_messages.append(_message)
        return chatgpt_messages

    
