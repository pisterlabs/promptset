import argparse
from typing import Optional
from dotenv import load_dotenv
import os
from tqdm import tqdm
import numpy as np

# torch
import torch

# init hugging face
from langchain import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, LLMMathChain

import json

load_dotenv()


def input_extraction(datapoint):
    extracted_input = {
        "question": datapoint['question'],
        "context": datapoint['context'][len("CONTEXT:"):].replace('=', '\n'),
    }
    return extracted_input

raw_example1 = {
    "question": "If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be?", 
    "program": "PROGRAM:=Q1: What is the volume of a single penny?=Q2: What is the thickness of the penny which is in the form of a hollow sphere now?=A1: 0.93 in**3=A2: 3.9e-9 in=Q2 -> A2 | F2=Q1 -> A1 | F1=P: Div (Q1, Q2)", 
    "answer": "2.3e+8 in**2", 
    "context": "CONTEXT:=F1: The volume of the penny is 0.93 in**3=F2: The thickness of the penny is 3.9e-9 in"
}

raw_example2 = {
    "question": "How long would it take to pluck each hair out of your body one at a time?", 
    "program": "PROGRAM:=Q1: How long does it take to pluck a single strand of hair?=Q2: How many hairs do we have on our body?=A1: 5 s=A2: 5e+6=Q2 -> A2 | F2=Q1 -> A1 | F1=P: Mul (Q1, Q2)", 
    "answer": "2.5e+7 s", 
    "context": "CONTEXT:=F1: It takes around 5 seconds to pluck a single strand of hair.=F2: The entire human body has 5e+6 hair follicles."
}

SYSTEM_PROMPT_STRING = """
    You are a helpful assistant tasked with answering estimation questions using provided information.
    You are provided with contextual information which contains facts that is relevant to answering the estimation question. 
    You should first write a mathematical formula to estimate the desired quantity, then use the context to fill in the variables in the formula.
    Lastly, you should evaluate the formula to get the final answer.
"""

INPUT_PROMPT = PromptTemplate(
                        input_variables=["context", "question"], 
                        template="""CONTEXT:{context}\n\nQUESTION: {question}"""
                    )

COMBINED_PROMPT = PromptTemplate(
    template=f"{SYSTEM_PROMPT_STRING}\n\n{INPUT_PROMPT.template}",
    input_variables=INPUT_PROMPT.input_variables,
)

if __name__ == '__main__':
    llm = OpenAI(temperature=0)
    math_chain = LLMMathChain(llm=llm)

    chain = math_chain | COMBINED_PROMPT
    response = chain.run(input_extraction(raw_example1))
