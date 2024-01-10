# Import necessary libraries
from openai import OpenAI
import pandas as pd

# Define the LLM
llm = OpenAI()

# Define the datasets
general_prompts = ["What is 1 + 1?", "What is the square root of 16?", "What is 5 * 5?"]
specific_prompts = ["Provide the numerical answer to 1 + 1", "Provide the numerical answer to the square root of 16", "Provide the numerical answer to 5 * 5"]

# Define what constitutes a "precise" response
def is_precise(response):
    return response.isnumeric()

# Run the experiment
responses_general = [llm.create_completion(prompt) for prompt in general_prompts]
responses_specific = [llm.create_completion(prompt) for prompt in specific_prompts]

# Analyze the results
precise_responses_general = sum([is_precise(response) for response in responses_general])
precise_responses_specific = sum([is_precise(response) for response in responses_specific])

# Calculate the proportion of precise responses
proportion_general = precise_responses_general / len(general_prompts)
proportion_specific = precise_responses_specific / len(specific_prompts)

# Compare the results
if proportion_specific > proportion_general:
    print("More specific prompts result in more precise responses.")
else:
    print("The specificity of the prompt does not significantly influence the precision of the response.")

# Document the results
report = f"""
Experiment Report:

Hypothesis: More specific prompts result in more precise responses from the LLM.

Experiment Design: We created two datasets of prompts - one general and one specific. We input each prompt into the LLM and recorded the responses. We then determined whether each response was precise according to our definition (a numerical answer).

Results: The proportion of precise responses for general prompts was {proportion_general}. The proportion of precise responses for specific prompts was {proportion_specific}.

Conclusions: {'More specific prompts result in more precise responses.' if proportion_specific > proportion_general else 'The specificity of the prompt does not significantly influence the precision of the response.'}

Future Research: We suggest further research to explore the reasons for these results. For example, it may be that the LLM is more likely to provide a precise response when the prompt is phrased in a certain way.
"""

# Print the report
print(report)