
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import os
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm=OpenAI(temperature=0.6)

#first chain: take in a problem, generate 5 hypotheses
llm = OpenAI(temperature=0.5)
template= """
You are a top scientist. Given the following {problem}, generate a numbered list containing five hypotheses for possible research directions. 
"""
prompt_template=PromptTemplate(input_variables=["problem"],template=template)
hypothesis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="hyplist")

#chain to choose from the best hypothesis
llm=OpenAI(temperature=0.5)
template2 = """
Choose the simplest hypothesis from this list: {hyplist}
"""
prompt_template2 = PromptTemplate(
    input_variables=["hyplist"], template=template2)
choice_chain = LLMChain(
    llm=llm, prompt=prompt_template2, output_key="choice")

# This is an LLMChain to create an experiment given a hypothesis.
llm = OpenAI(temperature=.7)
template3 = """You are a scientist. Given this hypothesis, it is your job to create a concrete, testable experiment to test the hypothesis. Give a numbered list of steps I should follow to do the experiment in my lab, being as specific as possible. 

Hypothesis: {choice}
Scientist: Here's an experiment:"""
prompt_template = PromptTemplate(
    input_variables=["choice"], template=template3)
experiment_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="experiment")

# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SequentialChain
full_chain = SequentialChain(
    chains=[hypothesis_chain, choice_chain,experiment_chain],
    input_variables=["problem"],
    # Here we return multiple variables
    output_variables=["hyplist","choice", "experiment"],
    verbose=True)

#TODO: this will become HTML input routed through flask app
problem = input("Enter a problem: ")
output = full_chain({"problem":problem})