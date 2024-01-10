from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import HuggingFaceHub
from getpass import getpass
import os
from dotenv import load_dotenv

load_dotenv()

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
 
repo_id = "HuggingFaceH4/zephyr-7b-alpha"
 
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.4, "max_length":64})

# Define the source and target languages or frameworks
source_language = "python"
target_language = "c++"

source_code = """
for i in range(12):
    print(i)
"""

# Define the PromptTemplates for each step in the migration process

parsing_template = """
analyse this {source_language} code: 
{source_code}
"""

generation_template = """
generate {target_language} code from the parsed {source_language} code using this information {parsed_info}
"""

testing_template = """
test the generated {target_language} code to ensure that it is functionally equivalent to the parsed {source_language} code.

Generated code:
{generated_code}
"""

parsing_prompt_template = PromptTemplate(
    input_variables=["source_language","source_code"], template=parsing_template
)
parsing_chain = LLMChain(
    llm=llm, prompt=parsing_prompt_template, output_key="parsed_info"
)

generation_prompt_template = PromptTemplate(
    input_variables = ["target_language", "source_language", "parsed_info"],
    template = generation_template
)

generation_chain = LLMChain(
    llm = llm, prompt = generation_prompt_template, output_key="generated_code"
)

# testing_chain = LLMChain(
#     llm = llm, prompt = testing_template, output_key = "tested_code"
# )

overall_chain = SequentialChain(
    chains = [parsing_chain, generation_chain],
    input_variables = ["source_language", "target_language", "source_code"],
    output_variables = ["generated_code"]
)


target_code = overall_chain({"source_language":source_language, "target_language": target_language, "source_code": source_code})

print(target_code['generated_code'])