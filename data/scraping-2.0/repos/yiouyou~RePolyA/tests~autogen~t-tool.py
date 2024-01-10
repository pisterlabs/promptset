import os
_RePolyA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(_RePolyA)

from repolya.toolset.tool_langchain import search_yfinance_news, search_youtube, search_ddg_news
from repolya.toolset.util import parse_intermediate_steps

from langchain.callbacks import get_openai_callback
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools


llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
tools = [
    # search_yfinance_news,
    # search_youtube,
    # search_ddg_news,
]
tools.append(load_tools(['ddg-search']))
# print(tools)

tool_name = ['ddg-search', 'google-search-results-json']
tools = load_tools(tool_name, llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
    max_iterations=10,
    max_execution_time=20,
    early_stopping_method="generate" # "force/generate"
)

_ask = f"分别使用搜索引擎'{tool_name}'搜索'meta为什么要开源llama2，有什么商业目的'，再综合搜索结果分条列出答案"
with get_openai_callback() as cb:
    _re = agent({"input": _ask})
    _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
    # print(_token_cost)
    _ans = _re["output"]
    _steps = f"{_token_cost}\n\n" + parse_intermediate_steps(_re["intermediate_steps"])
    print(_steps)
    print(f"\n\noutput: '{_ans}'")

