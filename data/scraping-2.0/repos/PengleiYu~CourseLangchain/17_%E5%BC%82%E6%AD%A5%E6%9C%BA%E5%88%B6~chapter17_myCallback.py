import asyncio
from typing import Optional, Union, Any, Dict, List
from uuid import UUID

from langchain.chat_models import ChatOpenAI
from langchain.schema import LLMResult, HumanMessage
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.schema.output import GenerationChunk, ChatGenerationChunk


class MySyncHandler(BaseCallbackHandler):

    def on_llm_new_token(self, token: str, *, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
                         run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        print(f'获取到token: {token}')


class MyAsyncHandler(AsyncCallbackHandler):
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID,
                           parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        print('正在获取数据')
        await asyncio.sleep(1)
        print('数据获取完毕')

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                         tags: Optional[List[str]] = None, **kwargs: Any) -> None:
        print('正在整理结果')
        await asyncio.sleep(1)
        print('结果整理完毕')


async def main():
    chat = ChatOpenAI(
        max_tokens=100,
        # streaming=True,
        # callbacks=[MySyncHandler(), MyAsyncHandler(), ],
        callbacks=[MySyncHandler(), ],
        verbose=True,
    )
    # result = await chat.agenerate([[HumanMessage(content='哪种花卉最适合生日？只简单说3种，不超过50字')]])
    result = chat.generate([[HumanMessage(content='哪种花卉最适合生日？只简单说3种，不超过50字')]])
    print(result)


asyncio.run(main())
