import os
import asyncio
from typing import Any, Optional, List, Dict
from uuid import UUID

import uvicorn
from fastapi import FastAPI, Body,Request
from fastapi.responses import StreamingResponse
from queue import Queue

from langchain_core.agents import AgentFinish, AgentAction
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_toolkits import create_sql_agent
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain.tools import Mark  # hypothetical tool for handling Markdown

os.environ["OPENAI_API_KEY"] = "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"
#
# return duckdb.execute("SELECT * FROM read_parquet('..//data//risk//*.parquet') WHERE lastUpdatedAt > ?",
#                       [lastUpdatedAt]).fetchdf()
# #

import duckdb
from threading import Thread, current_thread
import random

duckdb_con = duckdb.connect('..//data//risk.db')
duckdb_con.execute("""
    CREATE OR REPLACE TABLE risk AS SELECT * from read_parquet('..//data//risk//*.parquet')
""")
duckdb_con.close()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uri = 'duckdb:///..//data//risk.db'

# connect_args = {'read_only': True}
#
# CONN = create_engine(uri)

db = SQLDatabase.from_uri(
    database_uri = uri
    ,include_tables=["risk"],sample_rows_in_table_info=5)

# initialize the agent (we need to do this for the callbacks)
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
    model_name="gpt-3.5-turbo",
    streaming=True,  # ! important
    verbose=False,
    callbacks=[]  # ! important (but we will add them later)
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    output_key="output"
)

agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[],
    llm=llm,
    verbose=False,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    return_intermediate_steps=True
)

sql_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    extra_tools=[MarkdownTool],
    verbose=True,
    return_intermediate_steps=True,
    max_iterations=40,
    handle_parsing_errors=True

)

class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False

    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""


    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.content = outputs.get('output')
        for i in self.content.split('\n'):
            self.queue.put_nowait(i)
        self.final_answer = True
        self.content = ""
        self.done.set()

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        print(f":::::::Agent Log {finish.log}")
        self.content = finish.log
        for i in self.content.split('\n'):
            self.queue.put_nowait(i)
        self.queue.put_nowait('\r\n')
        self.final_answer = True


    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        print(f"Agent aksiyon::::: {action.log}")


async def run_call(query: str, stream_it: AsyncCallbackHandler,isChatAgent:bool=True):
    # assign callback handler
    if isChatAgent:
        agent.agent.llm_chain.llm.callbacks = [stream_it]
        await agent.acall(inputs={"input": query})
    else:
        # sql_agent.callback_manager = [stream_it]
        # sql_agent.callback_manager.ll.llm_chain.llm.callbacks = [stream_it]
        sql_agent.callbacks = [stream_it]
        res = await sql_agent.acall(inputs={"input": query},return_only_outputs=True)



# request input format
class Query(BaseModel):
    text: str

async def create_gen(query: str, stream_it: AsyncCallbackHandler,isChatAgent:bool=True):
    task = asyncio.create_task(run_call(query, stream_it,isChatAgent))
    async for token in stream_it.aiter():
        print(':::::final answer:::::',token)
        yield token
    await task


@app.post("/chat")
async def chat(
        query: Query = Body(...),
):
    stream_it = AsyncCallbackHandler()

    gen = create_gen(query.text, stream_it)
    return StreamingResponse(gen, media_type="text/event-stream")

#
# @app.post("/sqlchat")
# async def chatsql(
#         query: Query = Body(...),
# ):
#     stream_it = AsyncCallbackHandler()
#     gen =  create_gen(query.text, stream_it,isChatAgent=False)
#
#     return StreamingResponse(gen, media_type="text/event-stream")


@app.post("/sqlchat")
async def stream_chat(request:Request):
    msgs = await request.json()
    msgs = msgs["messages"]
    stream_it = AsyncCallbackHandler()
    gen = create_gen(msgs[-1], stream_it, isChatAgent=False)

    return StreamingResponse(gen, media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
    )