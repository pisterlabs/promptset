import openai
import numpy as np
from scipy import stats

# Define the Experiment
questions = ["What is 1 + 1?", "What is the square root of 4?", "What is 2 * 2?", "What is 10 - 2?", "What is 3 ^ 3?"]
refined_questions = ["Provide the numerical answer to 1 + 1", "Provide the numerical answer to the square root of 4", 
                     "Provide the numerical answer to 2 * 2", "Provide the numerical answer to 10 - 2", 
                     "Provide the numerical answer to 3 ^ 3"]

def score_response(response):
    return 1 if response.isdigit() else 0

# Run the Experiment
def get_responses(prompts):
    responses = []
    for prompt in prompts:
        response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=5)
        responses.append(response.choices[0].text.strip())
    return responses

original_responses = get_responses(questions)
refined_responses = get_responses(refined_questions)

# Score the Responses
original_scores = [score_response(response) for response in original_responses]
refined_scores = [score_response(response) for response in refined_responses]

# Analyze the Results
original_avg = np.mean(original_scores)
refined_avg = np.mean(refined_scores)

t_stat, p_val = stats.ttest_ind(original_scores, refined_scores)

# Interpret the Results
if p_val < 0.05 and refined_avg > original_avg:
    print("The hypothesis is supported.")
else:
    print("The hypothesis is not supported.")