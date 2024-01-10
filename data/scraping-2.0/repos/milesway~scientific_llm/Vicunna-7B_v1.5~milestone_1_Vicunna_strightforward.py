import os
import openai
import backoff 

completion_tokens = prompt_tokens = 0

openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

my_model = "vicuna-7b-v1.5"

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=my_model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def gpt_complete(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=my_model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs



import csv

dataset_file = '/home/milesway/research/scientific_llm/tree-of-thought-llm/src/tot/data/24/24.csv'

with open(dataset_file, newline='') as file:
    reader = csv.DictReader(file)
    problems = []

    for row in reader:
        # Split the "Puzzles" column to get the four numbers
        puzzle_numbers = row['Puzzles'].split()
        problems.append(puzzle_numbers)

total_correct = 0
current_loop = 0
total_loops = len(problems)

# Process and evaluate each problem
for problem in problems:
    # Combine the first four columns to form the prompt
    prompt = ' '.join(problem[:4])

    # Generate response from the model for the given prompt
    # Note: You need to replace 'llm.generate' with the correct function call
    model_response = gpt(prompt)
    eva = lambda x: "24" in x
    is_correct = eva (model_response) 
    total_correct += int(is_correct)

    # Print problem, response, and evaluation result
    print(f"\nProblem: {prompt}")
    print(f"\nModel Response: {model_response}")
    print(f"\nCorrect: {is_correct}")
    print(f"\n=============={current_loop + 1} / {total_loops}====================\n")

    current_loop += 1

# Calculate and display the overall accuracy
accuracy = total_correct / total_loops
print(f"Overall Accuracy: {accuracy:.6f}")