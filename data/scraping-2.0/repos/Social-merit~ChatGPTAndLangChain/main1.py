# Import necessary modules and classes
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

# Initialize argument parser and define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Initialize OpenAI instance
llm = OpenAI()

# Create a prompt template for generating code
code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="Write a very short {language} function that will {task}."
)

# Create a prompt template for generating test code
test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}"
)

# Create a language model chain for generating code
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

# Create a language model chain for generating test code
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

# Create a sequential chain that runs both the code and test chains
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]
)

# Execute the sequential chain and store the result
result = chain({
    "language": args.language,
    "task": args.task
})

# Print the generated code and test code
print(">>>>>> GENERATED CODE:")
print(result["code"])
print(">>>>>> GENERATED TEST:")
print(result["test"])
