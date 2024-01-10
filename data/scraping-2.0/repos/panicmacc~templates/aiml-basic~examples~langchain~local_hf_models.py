import os
#os.environ["LANGCHAIN_HANDLER"] = "langchain"
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
#from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools

#
# Testing local models
#
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# blenderbot

#model_id = 'facebook/blenderbot-1B-distill'
#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
#
#pipe = pipeline(
#        "text2text-generation",
#        model=model,
#        tokenizer=tokenizer,
#        max_length=100
#)

# openassistant pythia
model_id = 'OpenAssistant/oasst-sft-1-pythia-12b'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100
)


local_llm = HuggingFacePipeline(pipeline=pipe)

llm_chain = LLMChain(prompt=prompt, llm=local_llm)

question = "What is the capital of England?"

print(llm_chain.run(question))

#
# Agent testing stuff
#


#search = SerpAPIWrapper()
#tools = [
#    Tool(
#        name = "Terminal",
#        func=bash.run,
#        description="Executes commands in a terminal. Input should be valid commands, and the output will be any output from running that command."
#    ),
#    #Tool(
#    #    name = "Current Search",
#    #    func=search.run,
#    #    description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term."
#    #),
#]

#
# Working example of agent chain
#

#tools = load_tools(["terminal"])
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#llm=ChatOpenAI(temperature=0)
#agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
#agent_chain.run(input="hi, i am bob")

#
# Agent chain with Chat model (needs testing)
#

#from langchain.chat_models import ChatOpenAI
#from langchain.llms import OpenAI
#from langchain.schema import (
#    AIMessage,
#    HumanMessage,
#    SystemMessage
#)

#chat = ChatOpenAI(temperature=0)
#llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)
#
#res = llm("Tell me a joke.")
#
#print(res)


#chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])

#messages = [
#    SystemMessage(content="You are a helpful assistant that translates English to French."),
#    HumanMessage(content="Translate this sentence from English to French. I love programming.")
#]
#chat(messages)
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})


