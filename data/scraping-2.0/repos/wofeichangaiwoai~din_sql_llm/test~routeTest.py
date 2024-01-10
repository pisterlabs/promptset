from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
import torch

from langchain.schema.output_parser import BaseOutputParser, OutputParserException

from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import OpenAI
from pydantic import Extra

from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun, AsyncCallbackManager, CallbackManager,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from functools import lru_cache

from test.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from ubix.common.llm import get_llm


class DummyChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

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
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        print('\nDummyChain:' + '====='*10)
        print(f'inputs:{inputs}')
        print('inputs:' + '====='*10)
        prompt_value = self.prompt.format_prompt(**inputs)
        print(f'prompt_value: {prompt_value}')
        print('prompt_value:' + '====='*10)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        print('acall, DummyChain:'+'====='*10)
        print(f'inputs:{inputs}')
        print('inputs:' +'====='*10)

        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        print(f'prompt_value:{prompt_value}')
        print('prompt_value:'+'====='*10)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "my_custom_chain"


llm = get_llm()

from langchain.chains.router import MultiPromptChain

from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template,
        "chain": ''
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
        "chain":''
    },
]


destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = DummyChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")


from langchain.output_parsers.json import parse_and_check_json_markdown
def remove_after_last_a(input_str):
    last_a_index = input_str.find('}')
    if last_a_index != -1:  # 如果找到了 'A'
        result_str = input_str[:last_a_index+1]
    else:
        result_str = input_str

    first_a_index = result_str.find('{')
    if first_a_index != -1:
        return result_str[first_a_index:]
    else:
        return result_str  # 如果没有找到 'A'，返回原始字符串


from langchain.chains.router.llm_router import RouterOutputParser

class RouterOutputParserExt(RouterOutputParser):
    def parse(self, text: str) -> Dict[str, Any]:
        text = remove_after_last_a(text)
        print("RouterOutputParserExt" + "======"*10)
        print(text)
        print("RouterOutputParserExt" + "======"*10)
        return super().parse(text)


from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParserExt(),
)

print(f"router_template:{router_prompt.template}")

class LogHandler(BaseCallbackHandler):
    def on_text(
            self,
            text: str,
            **kwargs: Any,
    ) -> Any:
        print(f"======== Here is the route info =========")
        print(text)


router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)
chain.destination_chains = destination_chains
# chain.router_chain.llm_chain.prompt.output_parser = RouterOutputParserExt()
chain.callbacks = CallbackManager([LogHandler()])


print(chain.run("What is black body radiation?"))


"""
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python test/routeTest.py
"""