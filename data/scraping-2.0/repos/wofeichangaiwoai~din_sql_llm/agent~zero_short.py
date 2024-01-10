"""Create a ChatVectorDBChain for question/answering."""
import pickle

import langchain
from langchain import LlamaCpp
from langchain import OpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.agents import Tool
from langchain.cache import InMemoryCache
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager, CallbackManager
from functools import lru_cache

from ubix.chain.chain_kb_ubix import get_kb_chain
from ubix.chain.chain_sql_ubix import get_db_chain


def get_tools(llm):
    global vectorstore
    with open("./local/vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_kb_chain(llm, vectorstore)
    local_tool = Tool(
        name='Query information about Ubix Company',
        func=qa_chain.run,
        description='Query information about me and Ubix Company',
    )
    local_tool.return_direct = True

    db_tool = Tool(
        name='Query Info about Data',
        func=get_db_chain(llm).run,
        description='Query Info about Data in table, Hive or Spark'
    )
    db_tool.return_direct = True

    tools = [
             # local_tool,
             db_tool ,
             # , get_general_chain()
             ]
    return tools
# 
# def get_agent(websocket=None, question_handler=None, stream_handler=None):
#     tools = get_tools()
#     manager = AsyncCallbackManager([])
#     if True:
#         tracer = LangChainTracer()
#         # tracer.load_default_session()
#         manager.add_handler(tracer)
#     zero_shot_agent = initialize_agent(
#         agent="zero-shot-react-description",
#         tools=tools,
#         llm=llm,
#         verbose=True,
#         # callback_manager=manager,
#     )
#     return zero_shot_agent


@lru_cache() 
def get_agent_conversion():
    from langchain.memory import ConversationBufferMemory
    llm = get_default_llm()
    tools = get_tools(llm)
    manager = AsyncCallbackManager([])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    from langchain.prompts import MessagesPlaceholder
    chat_history = MessagesPlaceholder(variable_name="chat_history")
    
    agent_chain = initialize_agent(tools, OpenAI(temperature=0),
                                   agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                   verbose=True,
                                   memory=memory,
                                   callback_manager=manager,
                                   agent_kwargs = {
                                       "memory_prompts": [chat_history],
                                       "input_variables": ["input", "agent_scratchpad", "chat_history"]
                                   })
    return agent_chain


def get_answer(question, verbose=False):
    zero_shot_agent = get_agent_conversion()
    answer = zero_shot_agent({"input": question})
    if verbose:
        return answer
    else:
        return answer["output"]


@lru_cache()
def get_default_llm():
    n_gpu_layers = 80  # Metal set to 1 is enough.
    n_batch = 512  # Should be between 1 and n_ctx
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    langchain.llm_cache = InMemoryCache()
    llm = LlamaCpp(
        #model_path="/Workspace/yamada/pretrain/LLaMa-7B-GGML/llama-7b.ggmlv3.q4_0.bin",
        model_path="/Workspace/yamada/pretrain/Llama-2-13B-chat-GGML/llama-2-13b-chat.ggmlv3.q4_0.bin",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2500,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,
        cache=True
    )
    return llm

@lru_cache()    
def get_route_llm():
    # llm = LlamaCpp(
    #     # model_path="/Workspace/yamada/pretrain/LLaMa-7B-GGML/llama-7b.ggmlv3.q4_0.bin",
    #     model_path="/Workspace/yamada/pretrain/Llama-2-13B-chat-GGML/llama-2-13b-chat.ggmlv3.q4_0.bin",
    #     n_gpu_layers=80,
    #     n_batch=512,
    #     n_ctx=2500,
    #     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    #     callback_manager= CallbackManager([StreamingStdOutCallbackHandler()]),
    #     verbose=True,
    #     # cache=True
    # )
    llm = OpenAI(temperature=0),
    return llm


if __name__ == '__main__':
    zero_shot_agent = get_agent_conversion()
    # result = zero_shot_agent({"input": "Who is Ubix?"})
    result = zero_shot_agent({"input": "Could you help to count how many rows are there in the table sales_order_item?"})
    # result = zero_shot_agent({"input": "Hi, I'm felix"})
    # result = zero_shot_agent({"input": "Who am I"})


"""
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python agent/zero_short.py
"""


