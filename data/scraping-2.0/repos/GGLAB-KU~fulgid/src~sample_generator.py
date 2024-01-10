import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import tiktoken
import random

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "text-curie-001"
TEMPERATURE = 0.3
MAX_TOKENS = 100

goals = ["Bake a chocolate cake", "Watch a movie on a streaming service", "Be a successful student"]


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the nuhe example goalsmber of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


@retry(wait=wait_random_exponential(min=1, max=480), stop=stop_after_attempt(32))
def call_GPT(prompt):
    num_tokens = num_tokens_from_string(prompt, MODEL_NAME)
    openai.api_key = OPENAI_API_KEY
    response = openai.Completion.create(
        engine=MODEL_NAME,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS - num_tokens,
    )
    return response.choices[0].text.strip()


# goal generation
def get_goal_prompt():
    w1_list = ["Write 1 goal ", "Generate a single goal "]
    w2_list = ["on ", " regarding ", "relating to "]
    w3_list = ["a subject ", " a topic ", "an area "]
    w4_list = ["chosen at random, unrelated to the examples", " of your choice, different from the examples ",
               "different to the examples"]
    w1 = random.sample(w1_list, 1)[0]
    w2 = random.sample(w2_list, 1)[0]
    w3 = random.sample(w3_list, 1)[0]
    w4 = random.sample(w4_list, 1)[0]
    prompt = f"{w1}, {w2} {w3} {w4}.\n\n"
    prompt += "Example goals:"
    chosen_goals = random.sample(goals, 3)
    for goal in chosen_goals:
        prompt += goal + '\n'
    return prompt


# procedure generation, taken from the article
def generate_prompt_prefix():
    w1_list = ["For a given goal ", " Given a goal "]
    w2_list = [" write down ", " break down into ", "put down " "jot down "]
    w3_list = [" steps ", " subgoals ", "a list of steps ", " several steps ", " several subgoals ", " some steps ",
               " some small steps "]
    w4_list = ["to achieve the goal ", " for achieving the goal ", "to attain the goal "]
    w1 = random.sample(w1_list, 1)[0]
    w2 = random.sample(w2_list, 1)[0]
    w3 = random.sample(w3_list, 1)[0]
    w4 = random.sample(w4_list, 1)[0]
    prompt_prefix = f"{w1}, {w2} {w3} {w4}.\n\n"
    return prompt_prefix


def main():
    print("Generating dataset")
    goal_num = 3
    for i in range(goal_num):
        goals.append(call_GPT(get_goal_prompt()))
    for goal_index in range(goal_num, len(goals)):
        plan = call_GPT(
            generate_prompt_prefix() + " Steps should be in the form of a numbered list.\n Goal: " + goals[goal_index])
        with open('dataset/plan_%i.txt' % goal_index, 'w') as f:
            f.write("Goal: " + goals[goal_index] + "\n" + plan)
        print(goals[goal_index], '\n' + plan)


main()
