from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import asyncio
import tiktoken
from .. import my_logger
import time
from typing import Any, Dict, List, AsyncGenerator

tokenizer = tiktoken.get_encoding("cl100k_base")

class OpenAiLLM():
    '''
    Creates object for accessing openai's apis with Langchain.
    '''
    model_name = ''
    llm = None

    def __init__(
            self, 
            model_name: str
        ):
        '''

        :param model_name:
        '''
        self.model_name = model_name
        # max_tokens are not set below so by default inf from openAI side
        # can be set later using llm.max_tokens = n
        # streaming False by default can be changed with llm.streaming = True
        # request_timeout = None, max_retries = 6 default values
        llm = ChatOpenAI(
            temperature=0,
            model_name = model_name,
        )
        self.llm = llm
        self.llm_stream = None

    def chat_completion(
            self, 
            prompt
        ):
        return self.llm(prompt)
    
    async def async_generate(
            self, 
            prompt,
            input_
        ):
        chain = LLMChain(
            llm=self.llm, 
            prompt=prompt
        )
        res = await chain.arun(input_)
        if 'ind' in input_:
            return input_['ind'], res
        else:
            return res
    
    async def async_generate_stream(
        self,
        prompt,
        input_
    ) -> AsyncGenerator[str, None]:
        chain = prompt | self.llm
        async for token in chain.astream(input_):
            yield token.content
        
    async def generate_concurrently(
            self, 
            prompts,
            inputs
        ):
        tasks = [
                self.async_generate(
                prompt, 
                input_
            ) for prompt, input_ in zip(
                    prompts,
                    inputs
                )
            ]
        return await asyncio.gather(*tasks)
    
    def get_len_token(self, text):
        tokens = tokenizer.encode(str(text))
        num_ip_tokens = len(tokens)
        return num_ip_tokens
    
    def create_summary_lists_by_token_len(self, list_summaries):
        total_num_tokens = 0
        output_list = []
        text = ''
        for idx, summary in enumerate(list_summaries):
            num_tokens = self.get_len_token(summary)
            if num_tokens + total_num_tokens < 3000:
                text += f'Summary part {idx}:\n{summary}\n'
                total_num_tokens += num_tokens
            elif idx-2 == len(list_summaries):
                output_list.append(text)
                return output_list, list_summaries[-1]
            else:
                output_list.append(text)
                text = f'Summary part {idx}:\n{summary}\n'
                total_num_tokens = num_tokens
        if text:
            output_list.append(text)
        return output_list, ''
        

    async def get_intermediate_summary (self, contexts, chat_prompt):
        inputs = []
        for idx, context in enumerate(contexts):
            inputs.append({
                "text":context,
                "ind":idx
            })
        contexts_map = [chat_prompt]*len(inputs)
        intermediate_summary = await self.generate_concurrently(
            contexts_map,
            inputs
        )
        intermediate_summary.sort(
            key = lambda x:x[0]
        )
        intermediate_summary = [i[1] for i in intermediate_summary]
        return intermediate_summary

    async def custom_map_reduce_summary(
            self,
            contexts,
            system_message,
            map_prompt,
            reduce_prompt,
            is_first_call = True
    ):
        if not is_first_call:
            contexts, remaining_summary_part = self.create_summary_lists_by_token_len(contexts)
            if len(contexts) == 1 and not remaining_summary_part:
                human_message = HumanMessagePromptTemplate(prompt = reduce_prompt)
                chat_prompt = ChatPromptTemplate.from_messages(
                    [
                        system_message, 
                        human_message
                    ]
                )
                summary = await self.generate_concurrently(
                    [chat_prompt],
                    [{"text":contexts[0]}]
                )
                return summary
            else:
                human_message = HumanMessagePromptTemplate(prompt = reduce_prompt)
                chat_prompt = ChatPromptTemplate.from_messages(
                    [
                        system_message, 
                        human_message
                    ]
                )
                intermediate_summary = await self.get_intermediate_summary (
                    contexts, chat_prompt
                )
                if remaining_summary_part:
                    intermediate_summary.append(remaining_summary_part)
                return await self.custom_map_reduce_summary(
                    intermediate_summary,
                    system_message,
                    map_prompt,
                    reduce_prompt,
                    is_first_call = False
                )
        else:
            human_message = HumanMessagePromptTemplate(prompt = map_prompt)
            chat_prompt = ChatPromptTemplate.from_messages(
                [
                    system_message, 
                    human_message
                ]
            )
            intermediate_summary = await self.get_intermediate_summary (
                contexts, chat_prompt
            )
            return await self.custom_map_reduce_summary(
                intermediate_summary,
                system_message,
                map_prompt,
                reduce_prompt,
                is_first_call = False
            )

chatgpt_35 = OpenAiLLM('gpt-3.5-turbo')
gpt4 = OpenAiLLM('gpt-4')