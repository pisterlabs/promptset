from typing import List

from langchain.agents import AgentExecutor
from linebot.v3.messaging.models import TextMessage
from linebot.v3.messaging.models.message import Message

from src.interactor.dtos.event_dto import EventInputDto
from src.interactor.interfaces.repositories.agent_executor_repository import (
    AgentExecutorRepositoryInterface,
)
from src.interactor.use_cases.message.cor.handler_base import Handler


class DefaultHandler(Handler):
    def handle(
        self,
        input_dto: EventInputDto,
        repository: AgentExecutorRepositoryInterface,
        response: List[Message],
    ):
        try:
            agent_executor = self._get_agent_executor(input_dto, repository)
            result = agent_executor.run(input=input_dto.user_input)
            response.append(TextMessage(text=result))
        except Exception as e:
            print(e)
            response.append(TextMessage(text="出現錯誤啦！請稍後再試。"))
        finally:
            return response
