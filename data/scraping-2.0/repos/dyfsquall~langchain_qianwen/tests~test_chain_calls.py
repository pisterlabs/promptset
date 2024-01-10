from unittest.mock import patch
from unittest import TestCase

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel

from mock_data import (mock_call_response, mock_streaming_generator,
                       MOCK_CALL_RESPONSE, MOCK_STREAMING_RESPONSE)

import sys
import asyncio

# 添加路径到sys.path
LOCAL_PACKAGE_PATH = '../langchain_qianwen/'
if LOCAL_PACKAGE_PATH not in sys.path:
    sys.path.append(LOCAL_PACKAGE_PATH)

from langchain_qianwen import Qwen_v1


class TestQwenChain(TestCase):
    def test_chain_predict(self):
        llm = Qwen_v1(
            model_name="qwen-turbo",
            streaming=False,
        )

        with patch.object(llm, 'client') as mock_client:
            mock_client.call.return_value = mock_call_response()

            test_template = """
                我是一个测试 {name} 的模版
            """

            test_prompt_template = PromptTemplate(
                input_variables=["name"],
                template=test_template,
            )

            chain = LLMChain(
                llm=llm,
                prompt=test_prompt_template,
            )

            response = chain.predict(name="test")

            assert response == MOCK_CALL_RESPONSE

    def test_chain_stream_predict(self):
        llm = Qwen_v1(
            model_name="qwen-turbo",
            streaming=True,
        )

        with patch.object(llm, 'client') as mock_client:
            mock_client.call.return_value = mock_streaming_generator()

            test_template = """
                我是一个测试 {name} 的模版
            """

            test_prompt_template = PromptTemplate(
                input_variables=["name"],
                template=test_template,
            )

            chain = LLMChain(
                llm=llm,
                prompt=test_prompt_template,
            )

            response = chain.predict(name="test")

            assert response == MOCK_STREAMING_RESPONSE
    
    def test_chain_async_stream(self):
        handler = AsyncIteratorCallbackHandler()

        llm = Qwen_v1(
            model_name="qwen-turbo",
            streaming=True,
            callbacks=[handler],
        )

        with patch.object(llm, 'client') as mock_client:
            mock_client.call.return_value = mock_streaming_generator()

            chain = ConversationChain(
                llm=llm,
                verbose=True,
            )

            response = chain_return_async_final_response(chain, handler, "hello")
            assert response == MOCK_STREAMING_RESPONSE

    def test_lcel_parallelism(self):
        template1 = "给我讲个有关 {topic} 的笑话"
        template2 = "写一个简短的关于 {story} 的故事"

        prompt1 = PromptTemplate.from_template(template1)
        prompt2 = PromptTemplate.from_template(template2)

        llm = Qwen_v1(
            model_name="qwen-turbo",
        )
        with patch.object(llm, 'client') as mock_client:
            mock_client.call.return_value = mock_call_response()

            chain1 = prompt1 | llm
            chain2 = prompt2 | llm

            combined = RunnableParallel(joke_resp=chain1, story_resp=chain2)
            response_dict = combined.invoke({"topic": "UI小红帽", "story": "产品大灰狼"})

            assert response_dict["joke_resp"] == MOCK_CALL_RESPONSE
            assert response_dict["story_resp"] == MOCK_CALL_RESPONSE
        

def chain_return_async_final_response(chain, handler, content):
    async_response = asyncio.run(_async_response(chain, handler, content))
    return async_response
 

async def _async_response(chain, handler, content) -> str:
    async def _use_async_handler(chain, handler, content):
        asyncio.create_task(chain.apredict(input=content))
        return handler.aiter()

    response = ''
    async_gen = await _use_async_handler(chain, handler, content)
    async for i in async_gen:
        response += i

    return response