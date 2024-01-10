from abc import ABC
<<<<<<< HEAD
from langchain.chains.base import Chain
from typing import Any, Dict, List, Optional, Generator
from langchain.callbacks.manager import CallbackManagerForChainRun
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
import torch
import transformers


class ChatGLMLLMChain(BaseAnswer, Chain, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    # 相关度
    top_p = 0.4
    # 候选词数量
    top_k = 10
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10
    streaming_key: str = "streaming"  #: :meta private:
    history_key: str = "history"  #: :meta private:
    prompt_key: str = "prompt"  #: :meta private:
    output_key: str = "answer_result_stream"  #: :meta private:
=======
from langchain.llms.base import LLM
from typing import Optional, List
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult)


class ChatGLM(BaseAnswer, LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10
>>>>>>> bc552302e9189af332f5ee655bd70d9a2e35b4d9

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
<<<<<<< HEAD
    def _chain_type(self) -> str:
        return "ChatGLMLLMChain"
=======
    def _llm_type(self) -> str:
        return "ChatGLM"
>>>>>>> bc552302e9189af332f5ee655bd70d9a2e35b4d9

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
<<<<<<< HEAD
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.prompt_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Generator]:
        generator = self.generatorAnswer(inputs=inputs, run_manager=run_manager)
        return {self.output_key: generator}

    def _generate_answer(self,
                         inputs: Dict[str, Any],
                         run_manager: Optional[CallbackManagerForChainRun] = None,
                         generate_with_callback: AnswerResultStream = None) -> None:
        history = inputs[self.history_key]
        streaming = inputs[self.streaming_key]
        prompt = inputs[self.prompt_key]
        print(f"__call:{prompt}")
        # Create the StoppingCriteriaList with the stopping strings
        stopping_criteria_list = transformers.StoppingCriteriaList()
        # 定义模型stopping_criteria 队列，在每次响应时将 torch.LongTensor, torch.FloatTensor同步到AnswerResult
        listenerQueue = AnswerResultQueueSentinelTokenListenerQueue()
        stopping_criteria_list.append(listenerQueue)
=======
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
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

>>>>>>> bc552302e9189af332f5ee655bd70d9a2e35b4d9
        if streaming:
            history += [[]]
            for inum, (stream_resp, _) in enumerate(self.checkPoint.model.stream_chat(
                    self.checkPoint.tokenizer,
                    prompt,
<<<<<<< HEAD
                    history=history[-self.history_len:-1] if self.history_len > 0 else [],
                    max_length=self.max_token,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stopping_criteria=stopping_criteria_list
=======
                    history=history[-self.history_len:-1] if self.history_len > 1 else [],
                    max_length=self.max_token,
                    temperature=self.temperature
>>>>>>> bc552302e9189af332f5ee655bd70d9a2e35b4d9
            )):
                # self.checkPoint.clear_torch_cache()
                history[-1] = [prompt, stream_resp]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": stream_resp}
<<<<<<< HEAD
                generate_with_callback(answer_result)
            self.checkPoint.clear_torch_cache()
=======
                yield answer_result
>>>>>>> bc552302e9189af332f5ee655bd70d9a2e35b4d9
        else:
            response, _ = self.checkPoint.model.chat(
                self.checkPoint.tokenizer,
                prompt,
                history=history[-self.history_len:] if self.history_len > 0 else [],
                max_length=self.max_token,
<<<<<<< HEAD
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                stopping_criteria=stopping_criteria_list
=======
                temperature=self.temperature
>>>>>>> bc552302e9189af332f5ee655bd70d9a2e35b4d9
            )
            self.checkPoint.clear_torch_cache()
            history += [[prompt, response]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}
<<<<<<< HEAD

            generate_with_callback(answer_result)
=======
            yield answer_result

>>>>>>> bc552302e9189af332f5ee655bd70d9a2e35b4d9

