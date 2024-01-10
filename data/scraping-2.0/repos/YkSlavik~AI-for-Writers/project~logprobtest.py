import numpy as np
import os
import openai
import dotenv
from joblib import Memory
from typing import List
from itertools import permutations

# Configure your OpenAI API key
dotenv.load_dotenv()
openai.organization = "org-9bUDqwqHW2Peg4u47Psf9uUo"
openai.api_key = os.getenv("OPENAI_API_KEY")

memory = Memory("./joblib_cache", verbose=0)

@memory.cache
def get_response(full_request):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=full_request,
        temperature=0.7,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=10,
        echo=True
    )
    return response

def get_all_responses(text: str) -> List[str]:
    sentences = text.split('. ')
    sentences[-1] = sentences[-1].rstrip('.')  # Remove the last full stop
    all_permutations = list(permutations(sentences))

    combinations = []
    for p in all_permutations:
        combination = ' '.join([s + '.' for s in p[:-1]] + [p[-1] + '.'])
        combinations.append(combination)

    return combinations

def compute_log_probs(question_w_outline, original_text, response):
    start_point = len(question_w_outline)
    #print(start_point)
    start_index = response.choices[0].logprobs.text_offset.index(start_point)
    #print(start_index)
    len_original_text = len(original_text)
    #print(len_original_text)
    end_point = start_point + len_original_text - 1
    end_index = min(range(len(response.choices[0].logprobs.text_offset)), key=lambda i: abs(response.choices[0].logprobs.text_offset[i] - end_point))
    #print(end_index)

    total = 0
    for x in range(start_index, end_index + 1):
        total = total + response.choices[0].logprobs.token_logprobs[x]
    return total



def calculate_logprobs_from_text(text: str, question_w_outline: str):
    combinations = get_all_responses(text)
    combinations = list(set(combinations))  # Remove duplicates
    logprobs = []

    for i in range(len(combinations)):
        full_request = question_w_outline + "\n" + combinations[i]
        print(f"Full request:\n{full_request}\n")
        response = get_response(full_request)
        combination_logprob = compute_log_probs(question_w_outline, text, response)
        logprobs.append((combinations[i], combination_logprob))

    return logprobs

question_w_outline_1 = "Write a short essay given this outline:\n• Four\n• One\n• Three\n• Two"

text_1 = "One orange. Two oranges. Three oranges. Four oranges"

# New text and outline
text = "In my early years, I struggled with reading and writing. My parents, both avid readers, encouraged me to practice and helped me improve. As I grew older, I began to appreciate the power of language and its ability to communicate ideas, evoke emotions, and shape the world around us."

question_w_outline = "Write a short essay given this outline:\n• Struggling with reading and writing in the early years\n• Parents' encouragement and help\n• The growing appreciation for the power of language"

# Calculate logprobs and display results
logprob_results = calculate_logprobs_from_text(text, question_w_outline)

for i, (combination, logprob) in enumerate(logprob_results, start=1):
    print(f"Combination {i}:")
    print(combination)
    print(f"Logprob: {logprob}\n")
