import numpy as np
from scipy import stats
import openai

def specificity_score(prompt, response):
    if 'numerical answer' in prompt:
        try:
            float(response)
            return 1
        except ValueError:
            return 0
    else:
        return None

general_prompts = ['What is 1 + 1?', 'What is the capital of France?', 'Who wrote "To Kill a Mockingbird"?']
specific_prompts = ['Provide the numerical answer to 1 + 1', 'Provide the name of the capital city of France', 'Provide the name of the author who wrote "To Kill a Mockingbird"']

# Set your OpenAI API key
openai.api_key = 'your-api-key'

general_responses = [openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=60).choices[0].text.strip() for prompt in general_prompts]
specific_responses = [openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=60).choices[0].text.strip() for prompt in specific_prompts]

general_scores = [specificity_score(prompt, response) for prompt, response in zip(general_prompts, general_responses)]
specific_scores = [specificity_score(prompt, response) for prompt, response in zip(specific_prompts, specific_responses)]

general_scores = [score for score in general_scores if score is not None]
specific_scores = [score for score in specific_scores if score is not None]

general_avg_score = np.mean(general_scores)
specific_avg_score = np.mean(specific_scores)

print(f'Average specificity score for general prompts: {general_avg_score}')
print(f'Average specificity score for specific prompts: {specific_avg_score}')

t_stat, p_value = stats.ttest_ind(general_scores, specific_scores)
print(f'T-test p-value: {p_value}')

with open('experiment_results.txt', 'w') as f:
    f.write(f'General prompts: {general_prompts}\n')
    f.write(f'Specific prompts: {specific_prompts}\n')
    f.write(f'General responses: {general_responses}\n')
    f.write(f'Specific responses: {specific_responses}\n')
    f.write(f'General scores: {general_scores}\n')
    f.write(f'Specific scores: {specific_scores}\n')
    f.write(f'Average specificity score for general prompts: {general_avg_score}\n')
    f.write(f'Average specificity score for specific prompts: {specific_avg_score}\n')
    f.write(f'T-test p-value: {p_value}\n')