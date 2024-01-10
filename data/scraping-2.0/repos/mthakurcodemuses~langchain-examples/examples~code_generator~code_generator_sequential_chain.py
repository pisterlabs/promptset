import argparse

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.chains import SequentialChain

# Load environment variables
load_dotenv()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers", type=str, help="Task to be performed")
parser.add_argument("--language", default="python", type=str, help="Language to be used")
args = parser.parse_args()

# Language model
llm = OpenAI()

# Chain 1 - Code Generation
# Prompt
code_prompt = PromptTemplate(input_variables=["language", "task"],
                             template="Write a short code {language} function that will {task}")
# Chain
code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")

# Chain 2 - Test Code Generation
# Prompt
test_code_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}",
)
# Chain
test_code_chain = LLMChain(llm=llm, prompt=test_code_prompt, output_key="test_code")

# Chaining the chains in sequence
sequential_chain = SequentialChain(chains=[code_chain, test_code_chain],
                                   input_variables=["language", "task"],
                                   output_variables=["code", "test_code"])

# Run the sequential chain
sequential_chain_result = sequential_chain({"language": args.language,
                                            "task": args.task})

print("######################################### Generated code #########################################\n")
print(sequential_chain_result["code"])

print("\n######################################### Generated Test #########################################\n")
print(sequential_chain_result["test_code"])
