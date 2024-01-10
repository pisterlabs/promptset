import random
import re
import openai

# Prepare the test data
general_prompts = ["What is 1 + 1?", "What is the capital of France?", ...]  # Add 50 general prompts
specific_prompts = ["Provide the numerical answer to 1 + 1", "Provide the numerical answer to the square root of 16", ...]  # Add 50 specific prompts

# Execute the test
general_responses = [openai.Completion.create(engine="text-davinci-002", prompt=prompt) for prompt in general_prompts]
specific_responses = [openai.Completion.create(engine="text-davinci-002", prompt=prompt) for prompt in specific_prompts]

# Analyze the results
def categorize_responses(responses):
    numerical_responses = [response for response in responses if re.match(r'^-?\d+(?:\.\d+)?$', response.choices[0].text.strip())]
    return numerical_responses

general_numerical_responses = categorize_responses(general_responses)
specific_numerical_responses = categorize_responses(specific_responses)

general_percentage = len(general_numerical_responses) / len(general_prompts) * 100
specific_percentage = len(specific_numerical_responses) / len(specific_prompts) * 100

# Verify the hypothesis
if specific_percentage > 80:
    print("Hypothesis is verified.")
else:
    print("Hypothesis is not verified.")

# Document the results
with open("results.txt", "w") as f:
    f.write(f"General prompts: {general_prompts}\n")
    f.write(f"Specific prompts: {specific_prompts}\n")
    f.write(f"General responses: {general_responses}\n")
    f.write(f"Specific responses: {specific_responses}\n")
    f.write(f"General numerical responses: {general_numerical_responses}\n")
    f.write(f"Specific numerical responses: {specific_numerical_responses}\n")
    f.write(f"General percentage: {general_percentage}\n")
    f.write(f"Specific percentage: {specific_percentage}\n")