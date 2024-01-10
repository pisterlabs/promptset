# Import necessary libraries
import openai
import pandas as pd
from scipy.stats import chi2_contingency

# Define the test set of prompts and expected responses
test_set = {
    "What is 1 + 1?": "2",
    "Who was the first president of the United States?": "George Washington",
}

# Define the more specific versions of the prompts
specific_prompts = {
    "What is 1 + 1?": "Provide a one-word answer: What is 1 + 1?",
    "Who was the first president of the United States?": "Provide a one-word answer: Who was the first president of the United States?",
}

# Initialize the LLM
llm = openai.Completion.create(engine="text-davinci-002", max_tokens=60)

# Run the LLM with the original and specific prompts and record the responses
original_responses = {}
specific_responses = {}

for prompt, expected_response in test_set.items():
    original_responses[prompt] = llm.complete(prompt=prompt)
    specific_responses[specific_prompts[prompt]] = llm.complete(prompt=specific_prompts[prompt])

# Evaluate the precision of the responses
original_precision = [1 if response == expected_response else 0 for response, expected_response in zip(original_responses.values(), test_set.values())]
specific_precision = [1 if response == expected_response else 0 for response, expected_response in zip(specific_responses.values(), test_set.values())]

# Calculate the proportion of precise responses
original_proportion = sum(original_precision) / len(original_precision)
specific_proportion = sum(specific_precision) / len(specific_precision)

# Perform a chi-square test of independence to test the hypothesis
contingency_table = pd.crosstab(index=[original_precision, specific_precision], columns=["precision"])
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Report the findings
print(f"Proportion of precise responses for original prompts: {original_proportion}")
print(f"Proportion of precise responses for specific prompts: {specific_proportion}")
print(f"Chi-square test p-value: {p}")

# If the p-value is less than 0.05, we reject the null hypothesis and conclude that there is a significant difference in the proportion of precise responses
if p < 0.05:
    print("The proportion of precise responses is significantly different between the original and specific prompts.")
else:
    print("There is no significant difference in the proportion of precise responses between the original and specific prompts.")