PRO_CON_TOOL = {"desc": "Good for answering questions about a product's pros and cons.", "name":"pro_and_con"}
PRO_CON_CONST =\
{"inputs": ["chat_history", "question"],
 "outputs": {"pros": "a js array of the product's pros based on the chat history.",
             "cons": "a js array of the product's cons based on the chat history.",
            "overview": "What is your overview on the product."},
 'template': """You are a secondhand dealer and assessing the user's product. Based on your questions and user answers from the chat history.
 {chat_history}
 Please give your best answer.
 {format_instructions}
 Question: {question}."""}

from langchain.llms import BaseLLM
from langchain import LLMChain
import sys
import os
sys.path.append(f'{os.path.dirname(__file__)}/../..')
from botcore.utils.prompt_utils import build_prompt
from langchain.memory.chat_memory import BaseChatMemory

from langchain.tools import Tool

def build_pros_cons_chain(model: BaseLLM, memory: BaseChatMemory):
    """
    Chain is designed to answer questions about pros and cons.
    Input: chain({"question": question})
    """
    inputs = PRO_CON_CONST['inputs']
    outputs = PRO_CON_CONST['outputs']
    template = PRO_CON_CONST['template']
    prompt = build_prompt(inputs, outputs, template)
    chain = LLMChain(llm=model, verbose=True, prompt=prompt, memory=memory)
    return chain

def build_pros_cons_tool(model: BaseLLM, memory: BaseChatMemory):
    name = PRO_CON_TOOL['name']
    desc = PRO_CON_TOOL['desc']
    chain = build_pros_cons_chain(model, memory)
    func = lambda question: chain.run(question)
    tool = Tool.from_function(func=func, name=name, description=desc)
    return tool

