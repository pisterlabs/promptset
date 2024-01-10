RECYCLING_TIP_TOOL = {"desc": "Good for answering questions about recycling tips.", "name":"recycling_tip"}
RECYCLING_TIP_CONST =\
{"inputs": ["chat_history", "question"],
 "outputs": {"tips": "a js array of recycling actions for the product mentioned in the chat history. This array'length should be larger than 3.",
             "effect": "a helpful explanation for the advantages of recycling the product mentioned in the chat history."},
 'template': """You are a secondhand dealer and assessing the user's product. Based on your questions and user answers from the chat history.
 {chat_history}
 Given a question: {question}.
 Please give your best answer:
 {format_instructions}."""}

from langchain.llms import BaseLLM
from langchain import LLMChain
import sys
import os
sys.path.append(f'{os.path.dirname(__file__)}/../..')
from botcore.utils.prompt_utils import build_prompt
from langchain.memory.chat_memory import BaseChatMemory
from langchain.tools import Tool

def build_recycling_tip_chain(model: BaseLLM, memory: BaseChatMemory):
    """
    Chain is designed to answer questions about pros and cons.
    Input: chain({"question": question})
    """
    inputs = RECYCLING_TIP_CONST['inputs']
    outputs = RECYCLING_TIP_CONST['outputs']
    template = RECYCLING_TIP_CONST['template']
    prompt = build_prompt(inputs, outputs, template)
    chain = LLMChain(llm=model, verbose=True, prompt=prompt, memory=memory)
    return chain

def build_recycling_tip_tool(model: BaseLLM, memory: BaseChatMemory):
    name = RECYCLING_TIP_TOOL['name']
    desc = RECYCLING_TIP_TOOL['desc']
    chain = build_recycling_tip_chain(model, memory)
    func = lambda question: chain.run(question)
    tool = Tool.from_function(func=func, name=name, description=desc)
    return tool

