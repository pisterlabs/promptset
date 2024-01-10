from typing import Iterator, List, Optional, Mapping, Any, Tuple
from functools import partial
from langchain.schema.output import GenerationChunk
import torch
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoModel, AutoTokenizer
from configs.base_config import BASE_CONFIG


# ## chatglm-6B llm
class ChatGLM(LLM):

    model_path: str = BASE_CONFIG["llm_model_path"]
    max_length: int = 32768
    temperature: float = 0.1
    top_p: float = 0.5
    history: List[Tuple[str, str]] = []
    streaming: bool = True
    model: object = None
    tokenizer: object = None

    @property
    def _llm_type(self) -> str:
        return "chatglm3-6B"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_path": self.model_path,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history": [],
            "streaming": self.streaming
        }

    # 涉及到与chatbot前端交互，需要使用history实现多轮对话
    def _stream(self,
                prompt: str,
                stop: List[str] = None,
                run_manager: CallbackManagerForLLMRun = None,
                **kwargs: Any) -> Iterator[GenerationChunk]:
        event_obj = self.get_custom_event_object(**kwargs)
        text_callback = partial(event_obj.on_llm_new_token)
        temperature, top_p = self.change_params(**kwargs)
        index = 0

        for i, (resp, _) in enumerate(
                self.model.stream_chat(self.tokenizer,
                                       prompt,
                                       self.history,
                                       max_length=self.max_length,
                                       top_p=top_p,
                                       temperature=temperature)):
            if i == 0:
                self.history += [(prompt, resp)]
            else:
                self.history[-1] = (prompt, resp)
            text_callback(resp[index:])
            generation = GenerationChunk(text=resp[index:])
            index = len(resp)
            yield generation

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Must call `load_model()` to load model and tokenizer!")
        if self.streaming:
            event_obj = self.get_custom_event_object(**kwargs)
            resp = self.generate_resp(prompt, event_obj, **kwargs)
        else:
            resp = self.generate_resp(prompt, **kwargs)

        return resp

    # 不与chatbot交互的地方不需要使用history
    def generate_resp(self,
                      prompt,
                      event_obj: StreamingStdOutCallbackHandler = None,
                      **kwargs):
        resp = ""
        index = 0
        temperature, top_p = self.change_params(**kwargs)
        if event_obj:
            text_callback = partial(event_obj.on_llm_new_token)
            for i, (resp, _) in enumerate(
                    self.model.stream_chat(self.tokenizer,
                                           prompt, [],
                                           max_length=self.max_length,
                                           top_p=top_p,
                                           temperature=temperature)):
                text_callback(resp[index:])
                index = len(resp)
        else:
            resp, _ = self.model.chat(self.tokenizer,
                                      prompt, [],
                                      max_length=self.max_length,
                                      top_p=top_p,
                                      temperature=temperature)
        event_obj.on_chain_end(outputs={})
        return resp

    def load_model(self, model_name_or_path: str):
        if self.model is not None or self.tokenizer is not None:
            return
        if not model_name_or_path:
            self.model_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                       trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True).half().cuda().eval()

    def unload_model(self):
        if self.model is not None:
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._identifying_params:
                self.k = v

    def change_params(self, **kwargs):
        if "temperature" in kwargs:
            temperature = kwargs["temperature"]
        else:
            temperature = self.temperature
        if "top_p" in kwargs:
            top_p = kwargs["top_p"]
        else:
            top_p = self.top_p

        return temperature, top_p

    def get_custom_event_object(self, **kwargs):
        if "call_back_partial" in kwargs:
            obj = kwargs["call_back_partial"]
            return obj
        else:
            return StreamingStdOutCallbackHandler()
