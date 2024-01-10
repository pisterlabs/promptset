import logging
import os
import re
from queue import Queue
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define constants for evaluation criteria
MAX_EVALUATIONS = 11
EVALUATION_SCORE_THRESHOLD = 7

# Define templates for system and human prompts
decomposition_system_template = SystemMessagePromptTemplate.from_template(
    "You are an AI that decomposes complex code generation tasks into smaller, manageable sub-tasks. Please output the decomposed plan as a detailed, markdown numbered list of steps."
)
decomposition_human_template = HumanMessagePromptTemplate.from_template(
    "Given the complex code generation task: '{task}', please decompose it into a detailed, numbered list of sub-tasks."
)
generation_system_template = SystemMessagePromptTemplate.from_template(
    "In your capacity as an AI, your task is to generate code that aligns with a given set of instructions. While developing this code, you should take into account the requirements for readability (the code should be easy to understand), efficiency (the code should be optimized for performance), and correctness (the code should accurately fulfill the intended purpose)."
)
generation_human_template = HumanMessagePromptTemplate.from_template(
    "Based on the provided instruction: {step}, your task is to generate a piece of code. The resulting code should meet the following criteria: it should be readable, allowing other developers to easily understand its logic; it should be efficient, performing the task with minimal use of resources; and it should be correct, accurately fulfilling the instruction's purpose."
)
ranking_system_template = SystemMessagePromptTemplate.from_template(
    "As an AI, your role is to evaluate and rank multiple proposed code solutions based on a set of quality metrics. The ranking should be expressed as a list of scores in a descending order, where each score is a numerical value between 0 and 10. The scores should reflect considerations such as the code's readability (how easy it is for a human to understand), correctness (whether the code accomplishes what it intends to), efficiency (how optimally the code uses resources), and overall quality. Please present the results in the format 'score : n'."
)
ranking_human_template = HumanMessagePromptTemplate.from_template(
    "Your task is to evaluate and rank the following code sequences based on their quality scores. When performing the ranking, you should consider factors such as readability (is the code easy to comprehend?), correctness (does the code do what it's supposed to do?), efficiency (how optimally does the code use resources?), and overall quality. Please evaluate each piece of code and assign it a score between 0 and 10. \n\n{generated}\n\nOnce you've assessed each code, compile the scores in a descending order (highest to lowest) in the following format: 'score : n'."
)
generation_loop_system_template = SystemMessagePromptTemplate.from_template(
    "You are an AI that develops code by taking into account not only the current instruction but also the context of previous instructions and pieces of code. The generated code should be seen as an evolution of the past codes, in direct response to the given instruction. It should be efficient, readable, and above all, correct."
)
generation_loop_human_template = HumanMessagePromptTemplate.from_template(
    """Generate code for the following instruction: {step}. 

    This task is part of a larger sequence of coding instructions, and hence, you should take into account the context of previous instructions and codes when developing your solution. 

    The relevant information from the previous stages is as follows:

   
    """
)

# Initialize ChatOpenAI instances with different temperatures and the same model
decomposition_llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
generation_llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
ranking_llm = ChatOpenAI(temperature=0.4, model_name='gpt-3.5-turbo')

# Initialize LLMChain instances with different prompts
decomposition_chain = LLMChain(llm=decomposition_llm, prompt=ChatPromptTemplate.from_messages([decomposition_system_template, decomposition_human_template]))
generation_chain = LLMChain(llm=generation_llm, prompt=ChatPromptTemplate.from_messages([generation_system_template, generation_human_template]))
ranking_chain = LLMChain(llm=ranking_llm, prompt=ChatPromptTemplate.from_messages([ranking_system_template, ranking_human_template]))
generation_loop_chain = LLMChain(llm=generation_llm, prompt=ChatPromptTemplate.from_messages([generation_loop_system_template, generation_loop_human_template]))

# Define the task
task = "Write a variable impedance control for force feedback using ros2, webots, webots_ros2 and ros2_control."
j = 1

# Decompose the task into a markdown list
markdown_list = decomposition_chain.run(task)

# Define the regular expression pattern to match the list items 
pattern = r'(\d+)\.\s+(.*)'

# Compile the regex pattern once
regex = re.compile(pattern)  

# Use a default dict to avoid KeyError
from collections import defaultdict
steps = defaultdict(str)  

# Find all matches of the pattern in the markdown list
matches = regex.findall(markdown_list)  

# Convert the matches into a dictionary
for match in matches: 
    steps[int(match[0])] = match[1] 

# Define an empty dictionary to store the generated output for each step
generated = {}

# Generate the output for step 1 four times using the LLMChain
for i in range(1, 5):
    output = generation_chain.run(steps[j])
    generated[i] = output

# Convert dictionary to string with separator and indicators
generated_string = ""
for i, code in generated.items():
    generated_string += f"\n\n{'='*50}\nCode {i}:\n{code}"

# Pass the generated code sequences to the ranking_chain
ranking = ranking_chain.run(generated_string)

# Extract code indicators and scores from ranking string
pattern = r"Code (\d+):\s*(\d+\.?\d*)"
matches = re.findall(pattern, ranking)

# Store code indicators and scores in dictionary of lists
ranked = {float(match[1]): int(match[0]) for match in matches}

# Get the highest score 
highest_score = max(ranked.keys())

# Get the code(s) for the highest score
highest_codes = ranked[highest_score]

prev_code = ""
prev_instruction = ""
actual_instruction = ""

j=1
# Extract code indicators and scores from ranking string
pattern = r"Code (\d+):\s*(\d+\.?\d*)"
matches = re.findall(pattern, ranking)

# Store code indicators and scores in dictionary of lists
ranked = {float(match[1]): int(match[0]) for match in matches}

# Get the highest score 
highest_score = max(matches, key=lambda x: float(x[1]))

# Get the code(s) for the highest score
highest_code = generated[highest_score[0]]

highest_code_str = generated[highest_score]

prev_code = prev_code + highest_code_str
k = j
j = j + 1
prev_instruction = steps[k] + prev_instruction
actual_instruction = steps[j]

while j <= 8:
    highest_code_str_memory = f"\n\n{'='*50}\nHighest Code:\n{highest_code_str}\n\n{'='*50}\nPrevious Code:\n{prev_code}\n\n{'='*50}\nPrevious Instruction:\n{prev_instruction}n\n{'='*50}\nActual Instruction:\n{actual_instruction}"
    generated_new = generation_loop_chain.run(highest_code_str_memory)
    # Define an empty dictionary to store the generated output for each step
    generated_new = {}

    # Generate the output for step 1 four times using the LLMChain
    for i in range(1, 5):
        output = generation_loop_chain.run(highest_code_str_memory)
        generated_new[i] = output

    # Convert dictionary to string with separator and indicators
    generated_string = ""
    for i, code in generated_new.items():
        generated_string += f"\n\n{'='*50}\nCode {i}:\n{code}"

    # Pass the generated code sequences to the ranking_chain
    ranking = ranking_chain.run(generated_string)

    # Store code indicators and scores in dictionary of lists
    ranked = {float(match[1]): int(match[0]) for match in matches}

    # Get the highest score 
    highest_score = max(ranked.keys())

    # Get the code(s) for the highest score
    highest_score = ranked[highest_score]

    # Select just the first code 
    highest_score = highest_score[0]

    highest_code_str = generated_new[highest_score]

    prev_code = prev_code + highest_code_str
    k = j
    j = j + 1
    prev_instruction = steps[k] + prev_instruction
    actual_instruction = steps[j]

    highest_code_str_memory = f"\n\n{'='*50}\nHighest Code:\n{highest_code_str}\n\n{'='*50}\nPrevious Code:\n{prev_code}\n\n{'='*50}\nPrevious Instruction:\n{prev_instruction}n\n{'='*50}\nActual Instruction:\n{actual_instruction}"
    print("##########################################################")
    print(highest_code_str_memory)




