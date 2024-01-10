import openai
import numpy as np
from scipy import stats

# Step 1: Define the Directness Metric
def directness_metric(response, answer):
    response_words = response.split()
    answer_words = answer.split()
    direct_words = [word for word in response_words if word in answer_words]
    return len(direct_words) / len(response_words)

# Step 2: Create a Dataset
questions = [
    {"general": "What is 1 + 1?", "specific": "Provide the numerical answer for 1 + 1", "answer": "2"},
    {"general": "What is the capital of France?", "specific": "Name the city that is the capital of France", "answer": "Paris"},
    {"general": "Who wrote 'To Kill a Mockingbird'?", "specific": "Identify the author of the book 'To Kill a Mockingbird'", "answer": "Harper Lee"},
    # Add more questions here
]

# Step 3: Generate Responses
for question in questions:
    question['general_response'] = openai.Completion.create(engine="your-engine", prompt=question['general'], max_tokens=60)
    question['specific_response'] = openai.Completion.create(engine="your-engine", prompt=question['specific'], max_tokens=60)

# Step 4: Score the Responses
for question in questions:
    question['general_score'] = directness_metric(question['general_response']['choices'][0]['text'], question['answer'])
    question['specific_score'] = directness_metric(question['specific_response']['choices'][0]['text'], question['answer'])

# Step 5: Compare the Scores
general_scores = [question['general_score'] for question in questions]
specific_scores = [question['specific_score'] for question in questions]

t_stat, p_val = stats.ttest_rel(general_scores, specific_scores)

# Step 6: Analyze the Results
if p_val < 0.05:
    print("The specificity of the prompt influences the directness of the LLM's response.")
else:
    print("Other factors may be influencing the directness of the LLM's response.")

# Step 7: Report the Findings
# This step is subjective and depends on how you want to present your findings.