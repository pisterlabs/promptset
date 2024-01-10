from abc import ABC
from langchain.llms.base import LLM
from typing import Optional, List, Any
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult)
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)

class ChatGLM(BaseAnswer, LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.8
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        print(f"__call:{prompt}")
        response, _ = self.checkPoint.model.chat(
            self.checkPoint.tokenizer,
            prompt,
            history=[],
            max_length=self.max_token,
            temperature=self.temperature
        )
        print(f"response:{response}")
        print(f"+++++++++++++++++++++++++++++++++++")
        return response

    def generatorAnswer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False):
        history += [[]]
        for stream_resp, temhistory in self.checkPoint.model.stream_chat(
                self.checkPoint.tokenizer,
                prompt,
                history=history[-self.history_len:-1] if self.history_len > 1 else [],
                max_length=self.max_token,
                top_p=self.top_p,
                temperature=self.temperature
        ):
            # self.checkPoint.clear_torch_cache()
            history[-1] = [prompt, stream_resp]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": stream_resp}
            yield answer_result

