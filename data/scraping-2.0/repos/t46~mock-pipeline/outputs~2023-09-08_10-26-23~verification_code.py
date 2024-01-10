import openai
import numpy as np
from scipy.stats import chi2_contingency

# Assuming that the prompts have been collected and stored in two lists: general_prompts and specific_prompts.
general_prompts = [...]  # Replace with your general prompts
specific_prompts = [...]  # Replace with your specific prompts

# Running the Experiment
# Input each prompt into the LLM and record the output.
general_responses = [openai.Completion.create(engine="text-davinci-002", prompt=prompt) for prompt in general_prompts]
specific_responses = [openai.Completion.create(engine="text-davinci-002", prompt=prompt) for prompt in specific_prompts]

# Data Analysis
# Categorize the responses into numerical and non-numerical responses and calculate the proportions.
def categorize_responses(responses):
    numerical_responses = sum([1 for response in responses if response['choices'][0]['text'].isdigit()])
    non_numerical_responses = len(responses) - numerical_responses
    return numerical_responses, non_numerical_responses

general_numerical, general_non_numerical = categorize_responses(general_responses)
specific_numerical, specific_non_numerical = categorize_responses(specific_responses)

# Use a chi-square test for independence to compare the proportions.
contingency_table = np.array([[general_numerical, general_non_numerical], [specific_numerical, specific_non_numerical]])
chi2, p_value, _, _ = chi2_contingency(contingency_table)

# Hypothesis Testing
# Compare the p-value with the significance level.
significance_level = 0.05

if p_value < significance_level:
    print("Reject the null hypothesis. The specificity of a prompt does influence the type of response from the LLM.")
else:
    print("Do not reject the null hypothesis. The specificity of a prompt does not significantly influence the type of response from the LLM.")