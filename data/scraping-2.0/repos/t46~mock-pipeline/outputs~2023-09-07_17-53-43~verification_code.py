from openai import OpenAI
from collections import Counter
import string

# Define the sets of prompts
P_G = ["What is 1 + 1?", "Tell me about dogs.", "What is the weather like?"]
P_S = ["Provide the numerical answer to 1 + 1", "Describe the characteristics of a German Shepherd.", "What is the current temperature in New York?"]

# Initialize the OpenAI API
openai = OpenAI()

# Function to run the LLM and record responses
def run_llm(prompts):
    responses = []
    for prompt in prompts:
        response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100)
        responses.append(response.choices[0].text.strip())
    return responses

# Run the LLM with the prompts
responses_G = run_llm(P_G)
responses_S = run_llm(P_S)

# Function to calculate specificity score
def specificity_score(response):
    words = response.split()
    non_generic_words = [word for word in words if word not in string.punctuation]
    return len(non_generic_words)

# Calculate the specificity scores
scores_G = [specificity_score(response) for response in responses_G]
scores_S = [specificity_score(response) for response in responses_S]

# Compare the specificity scores
comparison = [s1 > s2 for s1, s2 in zip(scores_S, scores_G)]

# Analyze the results
proportion = comparison.count(True) / len(comparison)
print(f"Proportion of prompt pairs for which the score for the response to the prompt in P_S is higher than the score for the response to the corresponding prompt in P_G: {proportion}")

# Draw conclusions
if proportion > 0.5:
    print("The hypothesis is supported. Consider using more specific prompts to improve the specificity of the LLM's responses.")
else:
    print("The hypothesis is not supported. Consider other ways to improve the specificity of the LLM's responses.")