import json 
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import time
from annoy import AnnoyIndex
import operator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from getpass import getpass
import gradio as gr
import sys


def normalize_instruction(instruction):
    """
    Normalize an assembly instruction for comparison.
    This function should be expanded to cover more cases.
    """
    # Example normalization rules (can be expanded)
    instruction = instruction.replace("mov", "move")
    instruction = instruction.lower()
    return instruction

def compare_gadgets(gadget1, gadget2):
    """
    Compare two ROP gadgets to determine if they are rewrites of one another.
    """
    normalized_gadget1 = [normalize_instruction(instr) for instr in gadget1]
    normalized_gadget2 = [normalize_instruction(instr) for instr in gadget2]

    return normalized_gadget1 == normalized_gadget2

def read_file_lines_to_list(filename):
    """
    Reads all lines from a file and stores them in a list of strings.

    :param filename: The name of the file to read from.
    :return: A list containing all lines in the file as strings.
    """
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            # Stripping newline characters from each line
            lines = [line.strip() for line in lines]
        return lines
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


OPENAI_API_KEY = getpass("Please enter your OPEN AI API KEY to continue:")
## load the OPENAI LLM model
open_ai_key = OPENAI_API_KEY

gadget1 = read_file_lines_to_list(sys.argv[2])
gadget2 = read_file_lines_to_list(sys.argv[3])

few_shot_prompt = read_file_lines_to_list(sys.argv[1])

llm = OpenAI(openai_api_key= OPENAI_API_KEY, model_name= "gpt-3.5-turbo-16k")
template = """ Take a deep breath. 
  You are cybersecurity researcher. You are tasked with identifying whether two return oriented programming (ROP) gadgets are rewrites of one another. 
  Below are example GADGETS that are rewrites of one another. They are written in assembly language and are labelled REWRITE GADGETS and numbered.

  {few_shot_prompt}

  GADGET1

    {gadget1}

  GADGET2
    
    {gadget2}

  Complete the task below to identify whether the two GADGETS GADGET1 and GADGET2 are rewrites of one another based on HARD INSTRUCTIONS below

  HARD INSTRUCTIONS
    1. Check whether GADGETS are semantical REWRITES of one another even if they are syntactically different.
    2. Output a line by line comparison of the two GADGETS and highlight the differences between the two GADGETS.
    3. Provide a REASON for your result
    4. Output SHOULD STRICTLY FOLLOW JSON SCHEMA
    SCHEMA
    class Result
        is_rewrite: boolean
        reason: string
        comparison: string
    """
prompt = PromptTemplate(template=template, input_variables=["few_shot_prompt", "gadget1", "gadget2"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
try:
    output = llm_chain.run({'few_shot_prompt': few_shot_prompt, 'gadget1':gadget1, 'gadget2':gadget2})
    print (output)
except Exception as e:
    print (e)
    print ("Error in running the LLM chain")
