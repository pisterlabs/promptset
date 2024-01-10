from typing import List
from .evaluators import Evaluator
from .tool import Tool
import sys

try:
    from langchain.tools import BaseTool as Langchain_Tool
except:
    Langchain_Tool = None

try:
    from openai.types import FunctionDefinition as OpenAI_Tool
except:
    OpenAI_Tool = None

try:
    from transformers import Tool as HuggingfaceHub_Tool
except:
    HuggingfaceHub_Tool = None

try:
    from llama_index.tools import BaseTool as LlamaIndex_Tool
except:
    LlamaIndex_Tool = None


class ToolKit:
    def __init__(self, tools: List[Tool]):
        self.tool_list = tools

    def get_tools(self) -> List[Tool]:
        return self.tool_list

    def evaluate(self, evaluator: Evaluator) -> float:
        return evaluator.evaluate(self.get_tools())

    @classmethod
    def from_langchain_tools(self, tools: List[Langchain_Tool]):
        if Langchain_Tool == None:
            print("Need `langchain` library installed", file=sys.stderr)
            raise ImportError
        toolset = [
            Tool(
                name=t.name,
                description=t.description,
                signature_schema=t.get_input_schema().schema(),
            )
            for t in tools
        ]
        return self(tools=toolset)

    @classmethod
    def from_openai_functions(self, tools: List[OpenAI_Tool]):
        if OpenAI_Tool == None:
            print("Need `openai` library installed", file=sys.stderr)
            raise ImportError
        toolset = [
            Tool(name=t.name, description=t.description, signature_schema=t.parameters)
            for t in tools
        ]
        return self(tools=toolset)

    @classmethod
    def from_huggingfaceHub(self, tools: List[HuggingfaceHub_Tool]):
        if HuggingfaceHub_Tool == None:
            print("Need `transformers` library installed", file=sys.stderr)
            raise ImportError
        toolset = [Tool(name=t.name, description=t.description) for t in tools]
        return self(tools=toolset)

    @classmethod
    def from_llamaIndex(self, tools: List[LlamaIndex_Tool]):
        if LlamaIndex_Tool == None:
            print("Need `llama_index` library installed", file=sys.stderr)
            raise ImportError
        toolset = [
            Tool(
                name=t.metadata.name,
                description=t.metadata.description,
                signature_schema=t.metadata.fn_schema.schema(),
            )
            for t in tools
        ]
        return self(tools=toolset)
