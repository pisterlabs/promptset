import openai

from langchain.chains import LLMChain
from langchain.agents.agent import AgentExecutor
from typing import Union, Dict, List


class CoreEngineBase():
    def get_embeds(self, text: str):
        raise BaseException("Not implemented")

    def run(self, messages: List[Dict[str, str]]):
        raise BaseException("Not implemented")


class OpenAICoreEngine(CoreEngineBase):
    def __init__(self, model="gpt-3.5-turbo", embed_model="text-embedding-ada-002"):
        super().__init__()
        self.model = model
        self.embed_model = embed_model

    def get_embeds(self, text) -> List[float]:
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=self.embed_model)['data'][0]['embedding']

    def run(self, messages: List[Dict[str, str]]):
        completion = openai.ChatCompletion.create(
            model=self.model, messages=messages)
        return completion.choices[0].message.content


class LangchainCoreEngine(CoreEngineBase):
    def __init__(self, chain: Union[LLMChain, AgentExecutor]):
        super().__init__()
        self._chain = chain

    def run(self, messages: List[Dict[str, str]]):
        return self._chain.run(messages)
