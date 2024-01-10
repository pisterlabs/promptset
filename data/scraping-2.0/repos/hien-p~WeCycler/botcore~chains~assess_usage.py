ASSESS_USAGE_TOOL = \
        {"desc": "Good for answering questions about checking a product's usability.",\
        "name": "assess_usage"}

ASSESS_USAGE_CONST =\
{"inputs": ['question', "chat_history"],
 "outputs": {"useable": "Is the given product still useable.",
             "reason": "A reason why the product is useable or not useable.",
             "function": "Assess how well the given product still can function."},
 'template': """You are a secondhand dealer and assessing the user's product. Based on your questions and user answers from the chat history.
 {chat_history}
 Please give your best answer for the given question from the user.
 {format_instructions}
 Question: {question}."""}

from langchain.llms import BaseLLM
from langchain import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
import sys
import os
sys.path.append(f'{os.path.dirname(__file__)}/../..')
from botcore.utils.prompt_utils import build_prompt

from langchain.tools import Tool

def build_assess_usage_chain(model: BaseLLM, memory: BaseChatMemory):
    """
    Chain is designe
    Input: chain({"question": "Do you think that it will function well in the future?"})
    """
    inputs = ASSESS_USAGE_CONST['inputs']
    outputs = ASSESS_USAGE_CONST['outputs']
    template = ASSESS_USAGE_CONST['template']
    prompt = build_prompt(inputs, outputs, template)
    chain = LLMChain(llm=model, verbose=True, prompt=prompt, memory=memory)
    return chain

def build_assess_usage_tool(model: BaseLLM, memory: BaseChatMemory):
    name = ASSESS_USAGE_TOOL['name']
    desc = ASSESS_USAGE_TOOL['desc']
    chain = build_assess_usage_chain(model, memory)
    run_func = lambda question: chain.run(question)
    tool = Tool.from_function(func=run_func, name=name, description=desc)
    return tool
