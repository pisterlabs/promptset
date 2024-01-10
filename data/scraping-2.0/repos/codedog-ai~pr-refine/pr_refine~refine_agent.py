from abc import ABC, abstractmethod

from langchain import LLMChain

from codedog_sdk.chains.bool_chain import BoolChain
from codedog_sdk.chains.list_chain import ListChain
from pr_refine.utils import load_gpt35_llm


class RefineAgent(ABC):
    @abstractmethod
    def detect(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def gen_choices(self) -> list[str]:
        raise NotImplementedError


class CheckpointAgent(RefineAgent):
    template_d = """下面我将提供一份{topic}文档的片段，请帮我判定这段内容中是否包含{point}。
片段如下:
---
{{content}}
---"""

    template_g = """下面我将提供一份{topic}文档的片段，请根据这段内容，帮我预设几种可能的{point},
片段如下:
---
{{content}}
___"""

    template_s = """下面我将提供一份{topic}文档的多条{point}，请帮我总结为通用的一段话，内容如下:
---
{{content}}
___
总结:"""

    def __init__(self, topic: str, point: str, content: str):
        self.content = content
        self._chain_d = BoolChain.from_llm(
            load_gpt35_llm(),
            self.template_d.format(topic=topic, point=point),
        )
        self._chain_g = ListChain.from_llm(
            load_gpt35_llm(0.2),
            self.template_g.format(topic=topic, point=point),
            4,
        )
        self._chain_s = LLMChain.from_string(
            load_gpt35_llm(0.1),
            self.template_s.format(topic=topic, point=point),
        )

    def detect(self) -> bool:
        flag = self._chain_d.run(content=self.content)
        return flag

    def gen_choices(self) -> list[str]:
        problems = self._chain_g.run(content=self.content)
        return problems

    def summarize(self, items: list[str]) -> str:
        summary = self._chain_s.run(content="\n\n".join(items))
        return summary
