import ast
import copy
import json
import logging
import random
import time
from typing import List, Optional, Tuple

import langchain
import openai
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import BaseLanguageModel, LLMResult

from src.generators import LMGenerator
from src.utils.tracking_utils import TokensTracker

logger = logging.getLogger(__name__)
from langchain import OpenAI, LLMChain, PromptTemplate, FewShotPromptTemplate
from langchain.schema import (
    AIMessage,
    BaseMessage, ChatResult, Generation, ChatGeneration
)
from langchain.chat_models import ChatOpenAI
import asyncio


class CachedChatOpenAI(ChatOpenAI):
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        messages_prompt = repr(messages)
        if langchain.llm_cache:
            results = langchain.llm_cache.lookup(messages_prompt, self.model_name)
            if results:
                chat_result = ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=result.text)) for result in results],
                    llm_output=results[0].generation_info)
                return chat_result
        chat_result = super()._generate(messages, stop)
        if langchain.llm_cache:
            results = [Generation(
                text=gen.message.content,
                generation_info=chat_result.llm_output
            ) for gen in chat_result.generations]
            langchain.llm_cache.update(messages_prompt, self.model_name, results)
        return chat_result

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        messages_prompt = repr(messages)
        if langchain.llm_cache:
            results = langchain.llm_cache.lookup(messages_prompt, self.model_name)
            if results:
                chat_result = ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=result.text)) for result in results],
                    llm_output=results[0].generation_info)
                return chat_result
        max_retries = 5
        n_retries = 0
        while True:
            n_retries = n_retries + 1
            try:
                chat_result = await super()._agenerate(messages, stop)
                break
            except openai.error.OpenAIError as e: #  RateLimitError
                time.sleep(30)
                if n_retries == max_retries:
                    raise e


        if langchain.llm_cache:
            results = [Generation(
                text=gen.message.content,
                generation_info=chat_result.llm_output
            ) for gen in chat_result.generations]
            langchain.llm_cache.update(messages_prompt, self.model_name, results)
        return chat_result


class OpenAIGenerator(LMGenerator):
    def __init__(self, prompt=None, model='gpt3'):
        """

        :param prompt:
        :param model: either "gpt3" or "Chatgpt"
        """
        self.model_type = model
        self.lm_class: BaseLanguageModel = None
        if model == 'gpt3':
            self.gen_kwargs = {
                "n": 1,
                'temperature': 0.7,
                'model_name': 'text-davinci-003',
                "top_p": 1,
                "max_tokens": 1000
            }
            self.lm_class = OpenAI

        elif model in ['chatgpt', 'gpt4']:
            self.gen_kwargs = {
                "n": 1,
                'model_name': "gpt-3.5-turbo-0613" if model == 'chatgpt' else 'gpt-4',
                'temperature': 1,
                "top_p": 1,
                "request_timeout": 600,
                "max_retries": 0,
            }
            self.lm_class = CachedChatOpenAI
        else:
            raise NotImplementedError()
        self.batch_size = 50
        self.prompt = prompt
        self.total_tokens = 0

    def generate(self, inputs: List[dict], parallel=True, **gen_kwargs) -> List[List[str]]:
        _gkwargs = copy.deepcopy(self.gen_kwargs)
        _gkwargs.update(**gen_kwargs)
        if self.model_type == 'gpt3' and _gkwargs.get('n', 1) > 1:
            _gkwargs['best_of'] = _gkwargs['n']
        assert langchain.llm_cache is not None
        lm = self.lm_class(**_gkwargs)
        chain = LLMChain(llm=lm, prompt=self.prompt)
        ret = []
        for i in range(0, len(inputs), self.batch_size):
            in_batch = inputs[i:i + self.batch_size]
            if parallel:
                async def gen():
                    tasks = [chain.agenerate([ib]) for ib in in_batch]
                    ret_list = await asyncio.gather(*tasks)
                    for lm_out_i in ret_list:
                        logger.info(lm_out_i.llm_output)
                        TokensTracker.update(lm_out_i.llm_output, module=type(self).__name__)
                    return LLMResult(generations=[lm_out_i.generations[0] for lm_out_i in ret_list], )

                lm_output = asyncio.run(gen())
            else:
                lm_output = chain.generate(in_batch)
                logger.info(lm_output.llm_output)
                TokensTracker.update(lm_output.llm_output)
            ret.extend([[g.text for g in gen] for gen in lm_output.generations])
        return ret

    async def agenerate(self, inputs: List[dict], **gen_kwargs)-> List[List[str]]:
        _gkwargs = copy.deepcopy(self.gen_kwargs)
        _gkwargs.update(**gen_kwargs)
        if self.model_type == 'gpt3' and _gkwargs.get('n', 1) > 1:
            _gkwargs['best_of'] = _gkwargs['n']
        assert langchain.llm_cache is not None
        lm = self.lm_class(**_gkwargs)
        chain = LLMChain(llm=lm, prompt=self.prompt)
        tasks = [chain.agenerate([ib]) for ib in inputs]
        ret_list = await asyncio.gather(*tasks)
        for lm_out_i in ret_list:
            logger.info(lm_out_i.llm_output)
            TokensTracker.update(lm_out_i.llm_output, module=type(self).__name__)
            self.total_tokens += lm_out_i.llm_output.get('token_usage', {}).get('total_tokens', 0)
        lm_output = LLMResult(generations=[lm_out_i.generations[0] for lm_out_i in ret_list])

        ret = [[g.text for g in gen] for gen in lm_output.generations]
        return ret

    def print_ex(self, ex):
        print(self.prompt.format(**ex))


class SimplePromptOpenAIGenerator(OpenAIGenerator):
    def __init__(self, prompt_template: PromptTemplate, model='chatgpt'):
        if model == 'gpt3':
            prompt = prompt_template
        elif model in ['chatgpt', 'gpt4']:
            prompt = ChatPromptTemplate.from_messages([
                HumanMessagePromptTemplate(prompt=prompt_template)
            ])
        else:
            raise NotImplementedError
        super().__init__(prompt=prompt, model=model)




message_type_to_prompt_class = {
    'human' : HumanMessagePromptTemplate,
    'ai':  AIMessagePromptTemplate
}


class JSONItemGenerator:
    def postprocess_generation(self, gen: str, expected_items: int =None) -> List[dict]:
        """
        Takes a (potentially multi-line) string and turns it into a list of dicts
        """
        results = []

        for line in gen.split('\n'):
            if not line.strip(): continue
            line = line.strip(', ')
            line = line.strip(".")
            try:
                results.append(ast.literal_eval(line.replace('null', "None")))
            except:
                try:
                    results.append(json.loads(line))
                except:
                    continue


        if expected_items and len(results) != expected_items:
            if len(results) > expected_items:
                results = results[:expected_items]
            else:
                res = [{} for _ in range(expected_items)]
                for r in results:
                    res[r['I'] - 1] = r
                if any(res):
                    results = res
                else: # final resort
                    results = results + [{} for _ in range(expected_items - len(results))]
        return results

class FollowupPromptOpenAIGenerator(OpenAIGenerator):
    def __init__(self, prompt_template_list: List[Tuple[str, PromptTemplate]], model='gpt3'):

        if model == 'gpt3':
            if any(isinstance(i, FewShotPromptTemplate) for i in prompt_template_list[1:]):
                raise NotImplementedError("cannot handle template lists that have fewshot prompts after the first")
            if isinstance(prompt_template_list[0][1], FewShotPromptTemplate):
                combined_template = '\n\n'.join(template.template for (_, template) in prompt_template_list[1:])
                first_prompt: FewShotPromptTemplate = prompt_template_list[0][1]
                prompt = FewShotPromptTemplate(
                    examples=first_prompt.examples,
                    example_selector=first_prompt.example_selector,
                    example_prompt=first_prompt.example_prompt,
                    suffix=first_prompt.suffix + '\n' + combined_template,
                    input_variables = first_prompt.input_variables + PromptTemplate.from_template(combined_template).input_variables,
                    example_separator = first_prompt.example_separator,
                    prefix=first_prompt.prefix
                )
            else:
                def _get_template(t):
                    if isinstance(t, BaseMessagePromptTemplate):
                        return t
                    else:
                        return t.template

                combined_template = '\n\n'.join(template.template for (_, template) in prompt_template_list)
                prompt = PromptTemplate.from_template(combined_template)
        elif model in ['chatgpt', 'gpt4']:
            prompt = ChatPromptTemplate.from_messages([
                message_type_to_prompt_class[_type](prompt=template) for (_type, template) in prompt_template_list
            ])
        else:
            raise NotImplementedError
        super().__init__(prompt=prompt, model=model)
