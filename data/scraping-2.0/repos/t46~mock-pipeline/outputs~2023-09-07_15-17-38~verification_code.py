import random
import re
import openai

# Initialize the GPT3 model
gpt3 = openai.GPT3()

# Define the Dataset
dataset = []
for _ in range(1000):
    operation = random.choice(['+', '-', '*', '/'])
    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)
    dataset.append((num1, operation, num2))

# Define the Prompts
general_prompts = [f"What is {num1} {operation} {num2}?" for num1, operation, num2 in dataset]
specific_prompts = [f"Provide the numerical answer to {num1} {operation} {num2}." for num1, operation, num2 in dataset]

# Run the Experiment
general_responses = [gpt3.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=5).choices[0].text.strip() for prompt in general_prompts]
specific_responses = [gpt3.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=5).choices[0].text.strip() for prompt in specific_prompts]

# Analyze the Responses
def is_numerical(response):
    return bool(re.match("^[0-9]+$", response))

general_numerical = sum(is_numerical(response) for response in general_responses)
specific_numerical = sum(is_numerical(response) for response in specific_responses)

general_percentage = (general_numerical / len(general_responses)) * 100
specific_percentage = (specific_numerical / len(specific_responses)) * 100

# Compare the Results
hypothesis_supported = specific_percentage > 80

# Document the Results
report = f"""
Experiment Report:

Dataset: {dataset}

General Prompts: {general_prompts}
Specific Prompts: {specific_prompts}

General Responses: {general_responses}
Specific Responses: {specific_responses}

Percentage of numerical answers for general prompts: {general_percentage}%
Percentage of numerical answers for specific prompts: {specific_percentage}%

Hypothesis Supported: {hypothesis_supported}

Limitations: This experiment assumes that the LLM can solve all types of mathematical problems, which may not be the case. It also assumes that the LLM will always provide a numerical answer when asked, which may not be the case.

Suggestions for Future Research: Further research could investigate the types of problems that the LLM struggles with, or the types of prompts that are most likely to elicit a numerical response.
"""

# Print the report
print(report)