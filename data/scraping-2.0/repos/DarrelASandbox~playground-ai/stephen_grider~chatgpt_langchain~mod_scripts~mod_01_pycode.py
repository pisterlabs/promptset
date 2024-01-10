"""
This script demonstrates how to use language learning models (LLMs), specifically OpenAI's GPT, 
to automatically generate code and corresponding tests based on a specified task.
It utilizes the langchain library for chaining tasks and generating prompts for the LLM.

The script works as follows:
1. Parses command-line arguments to get the task and programming language.
2. Initializes a language model using OpenAI's GPT.
3. Defines prompt templates for code and test generation.
4. Sets up LLM chains for generating code and tests sequentially.
5. Executes the chains and outputs the generated code and its test.

This script is intended as a tutorial for beginners new to using LLMs for automated code generation.
"""
import argparse
from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

# Set up argument parsing for command-line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Initialize the language model, in this case, using OpenAI's GPT
llm = OpenAI()

# Define a prompt template for generating code
code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a short {language} function that will {task}",
)

# Define a prompt template for generating a test for the generated code
test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}",
)

# Set up the chain for generating code
code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")

# Set up the chain for generating a test for the code
test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")

# Combine the two chains sequentially
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"],
)

# Execute the chain with the provided command-line arguments
result = chain({"language": args.language, "task": args.task})

# Print the generated code and its test
print(">>>>>> GENERATED CODE:")
print(result["code"])

print(">>>>>> GENERATED TEST:")
print(result["test"])
