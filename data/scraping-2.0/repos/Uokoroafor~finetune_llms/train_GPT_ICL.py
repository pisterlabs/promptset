import random
import openai
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

import math

# Perform an in context regression task

# Define the prompt
# Numerical prompts

numbers = []
for number in range(0, 30):
    numbers.append(str(3 * number**3 + 2 * number**2))

# Number prompts are in groups of 4
number_contexts = [
    [numbers[i : i + 3], numbers[i + 3]] for i in range(0, len(numbers) - 3)
]

# Shuffle the number prompts
random.shuffle(number_contexts)

# Only allowed 20
# number_contexts = number_contexts[:10]

contexts = []
answers = []

for context in number_contexts:
    contexts.append(", ".join(context[0]))
    answers.append(context[1])

# Define the prompt
question_prompt = " What number comes next in the sequence "


# Define the function to generate the prompt
def generate_prompt(context, answer=""):
    prompt = ""
    if answer == "":
        prompt += "input-> " + question_prompt + context + "? Ans-> "
    else:
        prompt += "input-> " + question_prompt + context + "? Ans-> " + answer + "\n"
    return prompt


def generate_few_shot_prompt(contexts, answers, num_shots=3, current_index=0):
    prompt = ""

    for _ in range(num_shots):
        while True:
            i = random.randint(0, len(contexts) - 1)
            if i != current_index:
                break
        prompt += generate_prompt(contexts[i], answers[i])

    prompt += generate_prompt(contexts[current_index])

    return prompt


for num_shot in range(5):
    # Generate the prompts
    prompts = [
        generate_prompt(context, answer) for context, answer in zip(contexts, answers)
    ]
    few_shot_prompts = [
        generate_few_shot_prompt(contexts, answers, num_shots=num_shot, current_index=i)
        for i in range(len(contexts))
    ]

    answers_ = []
    for p in few_shot_prompts[:1]:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=p,
            max_tokens=5,
        )
        answers_.append(response.choices[0].text.strip())

    # Print the prompts and answers
    for i, prompt in enumerate(few_shot_prompts[:1]):
        print(f"Prompt {i + 1}:\n{prompt}")
        print(f"Answer {i + 1}:\n{answers_[i]}")
        print(f"Correct Answer:\n{answers[i]}")
        print()
