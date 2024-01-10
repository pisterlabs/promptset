import random
import re
from openai import OpenAI

# Initialize OpenAI
openai = OpenAI()

# Prepare test data
questions = [(f"What is {i} + {j}?", f"Provide the numerical answer to {i} + {j}", i+j) for i in range(1, 32) for j in range(1, 32)]
random.shuffle(questions)
questions = questions[:1000]  # Select 1000 questions

# Execute test
general_responses = []
specific_responses = []

for general_prompt, specific_prompt, _ in questions:
    general_response = openai.Completion.create(engine="text-davinci-002", prompt=general_prompt, max_tokens=5)
    specific_response = openai.Completion.create(engine="text-davinci-002", prompt=specific_prompt, max_tokens=5)
    general_responses.append(general_response.choices[0].text.strip())
    specific_responses.append(specific_response.choices[0].text.strip())

# Analyze results
def count_numerical_responses(responses):
    return sum(1 for response in responses if re.match("^\d+$", response))

general_numerical_count = count_numerical_responses(general_responses)
specific_numerical_count = count_numerical_responses(specific_responses)

general_percentage = general_numerical_count / len(general_responses) * 100
specific_percentage = specific_numerical_count / len(specific_responses) * 100

# Compare and conclude
if specific_percentage > general_percentage:
    print("The hypothesis is supported.")
else:
    print("The hypothesis is not supported.")

# Document results
with open("results.txt", "w") as f:
    f.write(f"General prompts: {general_percentage}% numerical responses\n")
    f.write(f"Specific prompts: {specific_percentage}% numerical responses\n")
    f.write("The hypothesis is supported." if specific_percentage > general_percentage else "The hypothesis is not supported.")