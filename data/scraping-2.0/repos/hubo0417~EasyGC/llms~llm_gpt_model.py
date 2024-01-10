import openai
from typing import Iterator, List, Mapping, Any
from functools import partial
from langchain.schema.output import GenerationChunk
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from configs.base_config import AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT
from langchain.chat_models import AzureChatOpenAI


class ChatGPT(LLM):
    max_length: int = 16384
    temperature: float = 0.1
    top_p: float = 0.5
    streaming: bool = True
    model: dict = None
    tokenizer: object = None
    history: List[dict] = [{
        "role": "system",
        "content": "你是一个人工智能助手，你为用户解决各种问题"
    }]

    @property
    def _llm_type(self) -> str:
        return "azure-gpt-4"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming
        }

    # 需要与chatbot交互的地方，需要使用History完成多轮对话
    def _stream(self,
                prompt: str,
                stop: List[str] = None,
                run_manager: CallbackManagerForLLMRun = None,
                **kwargs: Any) -> Iterator[GenerationChunk]:

        openai.api_type = self.model["api_type"]
        openai.api_base = self.model["api_base"]
        openai.api_key = self.model["api_key"]
        openai.api_version = self.model["api_version"]

        event_obj = self.get_custom_event_object(**kwargs)
        text_callback = partial(event_obj.on_llm_new_token)
        temperature, top_p = self.change_params(**kwargs)
        # index = 0
        self.history.append({"role": "user", "content": prompt})
        for i, resp in enumerate(
                openai.ChatCompletion.create(engine="gpt-35-turbo-16k",
                                             messages=self.history,
                                             stream=True,
                                             top_p=top_p,
                                             temperature=temperature)):
            delta = resp['choices'][0]['delta']
            resp = delta['content'].replace(
                '\n', '').strip() if 'content' in delta else ""

            if i == 0:
                self.history.append({"role": "assistant", "content": resp})
            else:
                self.history[-1] = {
                    "role": "assistant",
                    "content": f"{self.history[-1]['content']}{resp}"
                }
            text_callback(resp)
            generation = GenerationChunk(text=resp)
            # index = len(resp)
            yield generation

    def _call(self,
              prompt: str,
              stop: List[str] = None,
              run_manager: CallbackManagerForLLMRun = None,
              **kwargs: Any) -> str:
        event_obj = self.get_custom_event_object(**kwargs)
        resp = self.generate_resp(prompt, event_obj, **kwargs)

        return resp

    def generate_resp(self,
                      prompt,
                      event_obj: StreamingStdOutCallbackHandler = None,
                      **kwargs):
        openai.api_type = self.model["api_type"]
        openai.api_base = self.model["api_base"]
        openai.api_key = self.model["api_key"]
        openai.api_version = self.model["api_version"]
        resp = ""
        temperature, top_p = self.change_params(**kwargs)
        resp = openai.ChatCompletion.create(engine="gpt-35-turbo-16k",
                                            messages=[{
                                                "role": "user",
                                                "content": prompt
                                            }],
                                            stream=False,
                                            top_p=top_p,
                                            temperature=temperature)

        resp = resp['choices'][0]['message']['content'].replace('\n',
                                                                '').strip()
        event_obj.on_chain_end(outputs={"gpt_result": resp})
        return resp

    def load_model(self, **kwargs):
        if self.model is not None:
            return
        self.model = {}
        self.model["api_type"] = "azure"
        self.model["api_base"] = AZURE_OPENAI_ENDPOINT
        self.model["api_key"] = AZURE_OPENAI_KEY
        self.model["api_version"] = "2023-05-15"

    def unload_model(self):
        self.model = None

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
