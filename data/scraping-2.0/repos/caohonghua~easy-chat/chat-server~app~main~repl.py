from flask_socketio import emit
from flask import request
from langchain.schema import LLMResult
from .. import socketio
import log

logger = log.setup_logger(__name__)

repl_socket_cache = {}


@socketio.on('connect', namespace='/repl')
def connect():
    user_id = request.args.get('user_id')
    client_ip = request.remote_addr
    user_key = client_ip + '-' + user_id
    get_repl_agent(user_key)
    logger.debug(f'聊天websocket连接成功, 客户端：{user_key}')


@socketio.on('disconnect', namespace='/repl')
def disconnect():
    user_id = request.args.get('user_id')
    client_ip = request.remote_addr
    user_key = client_ip + '-' + user_id
    del repl_socket_cache[user_key]
    logger.debug(f'聊天websocket连接断开, 客户端：{user_key}')


@socketio.on('message', namespace='/repl')
def message(data):
    user_id = request.args.get('user_id')
    client_ip = request.remote_addr
    user_key = client_ip + '-' + user_id
    logger.debug(f'接受到聊天消息：{data}, 客户端：{user_key}')
    chain = get_repl_agent(user_key)
    chain.run(input=data, callbacks=[ReplCallbackHandler(user_key=user_key)])


from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, load_tools
from config import model_name


CHAT_TEMPLATE = """You are a Python Coder Expert, I have something to ask you. If the AI does not know the answer to a question, it truthfully says it does not know.This AI can provide answers in the corresponding language based on the language used by the questioner.

Human: {input}
AI:"""

def get_repl_agent(user_key):
    """获取python repl"""

    # 判断是否存在缓存
    if user_key in repl_socket_cache:
        return repl_socket_cache[user_key]
    # 构建chain
    llm = ChatOpenAI(temperature=0, streaming=True, model_name=model_name)
    tools = load_tools(['python_repl'], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=False)
    repl_socket_cache[user_key] = agent
    return agent

from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
import tiktoken

encoding = tiktoken.get_encoding('cl100k_base')

class ReplCallbackHandler(BaseCallbackHandler):

    def __init__(self, user_key) -> None:
        self.user_key = user_key
        self.tokens = 0

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, **kwargs: Any) -> Any:
        return super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, **kwargs)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        emit('message', '$$over$$')

    def on_llm_new_token(self, token: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        """发送websocket信息"""
        emit('message', token)

    def on_llm_error(self, error: Exception | KeyboardInterrupt, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        return super().on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, **kwargs: Any) -> Any:
        logger.debug(f'input: {inputs}')
        input = inputs['input']
        self.tokens += len(encoding.encode(input))
        scratchpad = inputs['agent_scratchpad']
        self.tokens += len(encoding.encode(scratchpad))

    def on_chain_error(self, error: Exception | KeyboardInterrupt, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        return super().on_chain_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        logger.debug(f'res: {outputs}')
        if 'text' in outputs:
            res = outputs['text']
            self.tokens += len(encoding.encode(res))
        if 'output' in outputs:
            res = outputs['output']
            self.tokens += len(encoding.encode(res))
        logger.debug(f'总消耗token: {self.tokens}')
