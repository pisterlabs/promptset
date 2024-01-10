## Chain Prompting

# Chain Prompting refers to the practice of chaining consecutive prompts, where the output of a previous prompt becomes the input of the successive prompt.

# To use chain prompting with LangChain, you could:

    # Extract relevant information from the generated response.
    # Use the extracted information to create a new prompt that builds upon the previous response.
    # Repeat steps as needed until the desired output is achieved.

# PromptTemplate class makes constructing prompts with dynamic inputs easier. This is useful when creating a prompt chain that depends on previous answers. 

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    llm=llm,
    model="text-davinci-003",
    temperature=0
)

# Prompt 1

template_question = """What is the name of the famous scientist who developed the theory of general relativity?
Answer: """
prompt_question = PromptTemplate(
    template=template_question,
    input_variables=[]
)

# Prompt 2

template_fact = """Provide a brief description of {scientist}'s theory of general relativity.
Answer: """
prompt_fact = PromptTemplate(
    input_variables=["scientist"],
    template=template_fact
)

# Create a LLMChain for the first prompt

chain_question = LLMChain(
    llm=llm,
    prompt=prompt_question
)

# Run the LLMChain for the first prompt with an empty dictionary

response_question = chain_question.run({})

# Extract the scientist's name from the response

scientist = response_question.strip()

# Create the LLMChain for the second prompt 

chain_fact = LLMChain(
    llm=llm,
    prompt=prompt_fact
)

# Input data from the second prompt 

input_data = {"scientst":scientist}

# Run the LLMChain for the second prompt 

response_fact = chain_fact.run(input_data)

print("Scientist : ", scientist)
print("Fact : ", response_fact)
