from typing import Any, Callable, Coroutine

from fastapi import Depends

from backend.schemas.agent import AgentRun
from backend.schemas.user import UserBase
from backend.agent.agent_service import AgentService
from backend.memory.memory import AgentMemory
from backend.agent.model_factory import create_model
from backend.setting import settings
from backend.tokenizer.dependencies import get_token_service
from backend.tokenizer.token_service import TokenService
from backend.agent.chatbot import OpenAIAgentService

from backend.agent.dependancies import get_agent_memory


def get_agent_service(
    validator: Callable[..., Coroutine[Any, Any, AgentRun]],
    streaming: bool = False,
    azure: bool = False,  # As of 07/2023, azure does not support functions
) -> Callable[..., AgentService]:
    # mock user
    user = UserBase(id="seed", name="seed")

    def func(
        run: AgentRun = Depends(validator),
        user: UserBase = user,
        # user: UserBase = Depends(get_current_user),
        agent_memory: AgentMemory = Depends(get_agent_memory),
        token_service: TokenService = Depends(get_token_service),
    ) -> AgentService:

        model = create_model(run.model_settings, user,
                             streaming=streaming, azure=azure)
        return OpenAIAgentService(
            model,
            run.model_settings,
            agent_memory,
            token_service,
            callbacks=None,
        )

    return func
