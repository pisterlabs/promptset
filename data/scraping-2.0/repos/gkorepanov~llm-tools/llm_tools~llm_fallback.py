from typing import (
    AsyncIterator,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from llm_tools.tokens import (
    TokenExpense,
    TokenExpenses,
)
from llm_tools.chat_message import OpenAIChatMessage

from llm_tools.errors import (
    should_fallback_to_other_model,
    MultipleException,
)
from llm_tools.llm_streaming import StreamingOpenAIChatModel
from llm_tools.llm_streaming_base import StreamingLLMBase



class StreamingModelWithFallback(StreamingLLMBase):
    def __init__(
        self,
        models: List[StreamingOpenAIChatModel],
        should_fallback_to_other_model: Callable[[Exception], bool] = \
            should_fallback_to_other_model, 
    ):
        self.models = models
        self.should_fallback_to_other_model = should_fallback_to_other_model
        self.exceptions = []
    
    async def stream_llm_reply(
        self,
        messages: List[OpenAIChatMessage],
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[Tuple[str, str]]:
        self.exceptions = []
        for model in self.models:
            try:
                async for completion, token in model.stream_llm_reply(messages, stop):
                    yield completion, token
            except Exception as e:
                if self.should_fallback_to_other_model(e):
                    self.exceptions.append(e)
                    continue
                else:
                    raise
            else:
                break
        else:
            if len(self.exceptions) == 1:
                raise self.exceptions[0]
            else:
                raise MultipleException(self.exceptions) from self.exceptions[-1]

    @property
    def succeeded(self) -> bool:
        return any(model.succeeded for model in self.models)

    def get_tokens_spent(
        self,
        only_successful_trial: bool = False,
    ) -> TokenExpenses:
        
        if not self.succeeded and only_successful_trial:
            raise ValueError("Cannot get tokens spent for unsuccessful trial")
            
        if only_successful_trial:
            first_successful_model = next(model for model in self.models if model._succeeded)
            return first_successful_model.get_tokens_spent(only_successful_trial)
        else:
            return sum(
                (
                    model.get_tokens_spent(only_successful_trial)
                    for model in self.models
                ),
                TokenExpenses()
            )
