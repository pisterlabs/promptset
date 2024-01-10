import asyncio
from pathlib import Path
from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms.utils import LLMType

from xecretary_core.utils.utility import create_llm


def get_query_engine(index_dir: str, llm: Optional[LLMType]) -> BaseQueryEngine:
    storage_dir = Path("./storage") / index_dir
    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    index = load_index_from_storage(storage_context, index_id=index_dir)
    llm_predictor = LLMPredictor(llm=llm)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    return index.as_query_engine(service_context=service_context)


def create_user_context_predictor_tool(
    name: str,
    index_dir: Optional[str],
    data_source: Optional[str],
    llm_name: Optional[str],
) -> BaseTool:
    if not (llm_name and index_dir):
        raise ValueError(
            f"Config error : set 'llm_name' and 'index_dir' for tool {name}"
        )

    llm = create_llm(llm_name)
    query_engine = get_query_engine(index_dir, llm)

    class UserContextPredictorTool(BaseTool):
        name: str = f"user_context_from_{data_source}"
        description: str = f"""
        ユーザーしか知らない知識を{data_source}から取得するツール
        """

        def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            response = query_engine.query(query)
            return str(response)

        async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            return str(
                await asyncio.get_event_loop().run_in_executor(
                    None, query_engine.query, query
                )
            )

    return UserContextPredictorTool()
