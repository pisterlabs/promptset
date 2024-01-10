

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI, ChatMessage
from llama_index.tools import BaseTool, FunctionTool
from typing import Sequence, AnyStr, Dict, Any, List


class LlamaAgent(object):
    def __init__(self, tools: List[BaseTool], llm: OpenAI) -> None:
        self._llm = llm
        self._chat_history = []
        self._agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)

    def reset(self) -> None:
        self._chat_history.clear()

    def chat(self, role: AnyStr, prompt: AnyStr) -> AnyStr:
        history = self._chat_history
        history.append(ChatMessage(role=role, content=prompt))
        answer = self._agent.chat(prompt)
        history.append(ChatMessage(role='assistant', content=str(answer)))
        return str(answer)


def multiply(a: int, b: int) -> int:
    return a + b


def exp(a: int, b: int) -> int:
    import math
    return int(math.exp(a)) + b


if __name__ == '__main__':
    import openai
    from llama_index.agent import OpenAIAgent

    """
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=[{'role':'user', 'content': 'what is 34 * 12'}],
        temperature=0,
        max_tokens=64
    )
    a = response
    """

    input_tools = [
        FunctionTool.from_defaults(fn=multiply),
        FunctionTool.from_defaults(fn=exp)
    ]

    open_ai = OpenAI(temperature=0, model='gpt-3.5-turbo-0613', verbose=True)
    agent = OpenAIAgent.from_tools(tools=input_tools, llm=open_ai)
    response = agent.chat('What is exponential 3')
    print(response)
    
    


