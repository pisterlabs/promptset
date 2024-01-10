from abc import ABC
from langchain.llms.base import LLM
from typing import Optional, List
from src.models.loader import LoaderCheckPoint
from src.models.base import (BaseAnswer, AnswerResult)
from transformers.generation import GenerationConfig


class Baichuan2(BaseAnswer, LLM, ABC):
    checkPoint: LoaderCheckPoint = None
    history_len: int = 5

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint
        self.checkPoint.model.generation_config = GenerationConfig(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            user_token_id=195,
            assistant_token_id=196,
            max_new_tokens=4096,
            temperature=0.3,
            top_k=5,
            top_p=0.85,
            repetition_penalty=1.05,
            do_sample=True,
        )

    @property
    def _llm_type(self) -> str:
        return "Baichuan2"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"__call:{prompt}")
        resp = self.checkPoint.model.chat(
            self.checkPoint.tokenizer,
            [{"role": "user", "content": prompt}],
            stream=False,
        )
        print(f"response:{resp}")
        print(f"+++++++++++++++++++++++++++++++++++")
        return resp


    def generatorAnswer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False):
        messages = []
        for x in history[-self.history_len:]:
            messages.append({"role": "user", "content": x[0]})
            messages.append({"role": "assistant", "content": x[1]})
        messages.append({"role": "user", "content": prompt})

        if streaming:
            history += [[]]
            position = 0
            for resp in self.checkPoint.model.chat(
                self.checkPoint.tokenizer,
                messages,
                stream=True,
            ):
                print(resp[position:], end="", flush=True)
                position = len(resp)
                history[-1] = [prompt, resp]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": resp}
                yield answer_result
            self.checkPoint.clear_torch_cache()
        else:
            resp = self.checkPoint.model.chat(
                self.checkPoint.tokenizer,
                messages,
                stream=False,
            )
            print(f"resp: {resp}")
            self.checkPoint.clear_torch_cache()
            history += [[prompt, resp]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": resp}
            yield answer_result
