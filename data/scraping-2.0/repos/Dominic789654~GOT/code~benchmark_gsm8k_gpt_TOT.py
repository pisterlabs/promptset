import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_chain,
    wait_fixed,
)  # for exponential backoff
import datasets
from tqdm import trange
from datasets import load_dataset, load_from_disk
import requests as requests
from api_pool import api_pool
import re
import time
import numpy as np
import random
from utils  import *

key_pool = api_pool
print(f"Number of api keys {len(key_pool)}")

use_chat_api = True
api_model = "gpt-3.5-turbo-0301"

# Global variables for token counts
global_total_tokens = 0
global_prompt_tokens = 0
global_completion_tokens = 0


def openai_api_call_handler(prompt, max_tokens, temperature, k=1, stop=None):
    global global_total_tokens, global_prompt_tokens, global_completion_tokens

    # while True:
    try:
        if use_chat_api:
            messages = [{"role": "user", "content": prompt}]
            response = completion_with_backoff(
                {
                    "model": api_model,
                    "messages": messages,
                    "temperature": temperature,
                }
            )
        else:
            response = completion_with_backoff(
                engine=api_model,
                prompt=prompt,
                n=k,
                max_tokens=max_tokens,
                stop=stop,
                temperature=temperature,
            )

        with open("openai.logs", "a") as log_file:
            log_file.write(
                "\n" + "-----------" + "\n" + "Prompt : " + prompt + "\n"
            )
        global_total_tokens += response["usage"]["total_tokens"]
        global_prompt_tokens += response["usage"]["prompt_tokens"]
        global_completion_tokens += response["usage"]["completion_tokens"]

        return response

    except Exception as e:
        # Print the error message if an exception occurs
        error_message = str(e)
        print(error_message)


def openai_choice2text_handler(choice):
    if use_chat_api:
        text = choice["message"]["content"]
    else:
        text = choice.text.strip()
    return text


def generate_text(prompt, k):
    if use_chat_api:
        thoughts = []
        for _ in range(k):
            response = openai_api_call_handler(prompt, 400, 1.1, k)
            text = openai_choice2text_handler(response["choices"][0])
            thoughts += [text]
        return thoughts
    else:
        response = openai_api_call_handler(prompt, 300, 1.1, k)
        thoughts = [openai_choice2text_handler(choice) for choice in response.choices]
        return thoughts


def ranking(prompt, question, past):
    # ranks = []
    # for i in range(len(prompt)):
    comparison_prompt = f"""
  To achieve the following goal: '{question}', and based on the current steps taken towards solving the problem {past}
  pessimistically value the below mentioned step and choose one of the follwing options that will be the best option towards the goal.
  Return the exact same chosen option, dont change or format it.
  The options to choose from \n
  {prompt}\n

  NOTE:
  1) Evaluate all the options and choose the option which is the best direction for the next step to move based on the past solution we have found till now. Dont choose the output that jumps to the result directly.
  2)MAKE SURE YOU DONT CHOOSE THE OPTION THAT HAS A SIMILAR MEANING (STEP) TO WHAT IS ALREADY THERE IN THE PAST SOLUTION ARRAY.

  DO NOT RETURN ANYTHING ELSE JUST THE OPTION THAT IS THE BEST NEXT STEP, NO EXPLANATION FOR THE CHOICE
  """
    a = generate_text(comparison_prompt, 1)
    return a


def parse_output_options(output):
    # output = output.split("Output")[1:]
    # output = " ".join(output).strip()
    output = output.split("\n")
    return output


"""# Having multiple GPT instances (num thoughts =k) each with multiple thoughts"""

initial_promp_temp = f"""
Imagine you are trying to solve a math problem with a step-by-step approach. At each step, you should propose a single next step to solve the problem involving a single arithmetic option. If there are multiple options for how to proceed, you should generate up to 3 options.

The format of the problem is as below, follow this format only
Input: XXXX
Steps taken so far: YYYY
Output: ZZZZ

NOTE: The options should not be sequential or connected with each other, each option should be in a way that it can be evaluated independently. Dont jump to the result directly.
IMPORTANT: MAKE SURE NOT TO HAVE THE DIRECT ANSWER IN YOUR POSSIBLE STEPS OUTPUT, JUST MAKE ONE STEP AT A TIME.
Solved Example:

Example 1
Input: "Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, a pound of cream cheese that cost half the price of the cheddar cheese, and a pack of cold cuts that cost twice the price of the cheddar cheese. How much does he spend on the ingredients?"

Steps take so far: [Calculate the price of cheddar cheese which is $10 (given)]


Output: Possible independent steps:
1) Calculate the price of cold cuts which is; Solving =  2*10 = $20.
2)Calculate the price of cream cheese which is; Solving = 10/2 = $5 per pound.

Example 2
Input: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

Steps taken so far: [None]

Output: Possible next steps:
1) Convert the minutes of babysitting to hours; Solving = 50/60 = 0.833
2) Convert the wage per hour to wage per minute; Solving = 12/60 = $0.2 per minute

Example 3
Input: "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"

Steps taken so far: [step 1: Number of letter written to 1 friend in a week = 2 as he writes twice a week, step 2: Number of letter written to 2 friends in a week ; Solving = 2*2 = 4 letters a week.,step 3: Number of letters written to both the friends in a year; Solving = 4*52 = 208 letters.]

Output: Possible next steps:
1) Number of pages written to both the friends in a year. This will be our final solution; Solving = 208*3 = 624 pages.


Now give the possible steps for the below question, Dont directly give the final answer to the question just solve that independant step arithmetically.

Input:

"""

output_string = " \n Output: Possible independent steps:"

summary_question_prompt = """
Given the question, try to give the final goal of the question in less than 10 words
Question:

"""

predict_prompt = """
Using only the steps provided below and the summary of the question, try to predict the final answer for the question and output just the final answer number, dont output any text. Use only the knowledge provided in the steps below.
Question Summary -

"""

"""## k=1 Max steps 5"""

# Parameters
questions_big = []
status_big = []


max_steps = 5
k = 2
pred = []
true = []
num_questions_to_solve = 1319
correct = 0
wrong = 0
total = 0

# dataset = load_dataset("gsm8k", "main")
dataset = load_from_disk("./data/gsm8k")
# dataset['test'] = dataset['test'].select(range(10))
for questions_number in trange(num_questions_to_solve):
    status = ["None"]

    question = dataset["test"]["question"][questions_number]
    true_answer = float(
        dataset["test"]["answer"][questions_number]
        .split("####")[-1]
        .strip()
    )
    # breakpoint()
    for i in range(max_steps):
        layer_options = []
        print("*****************NEW STEP*****************")
        print(f"The status array is {status} \n\n")
        initial_promp = (
            initial_promp_temp
            + str(question)
            + str("\n Steps taken so far:")
            + str(status)
            + output_string
        )
        out = generate_text(initial_promp, k)
        for j in range(k):
            print(f"######## This is the thought from instance number {j} ##########")
            outputs = parse_output_options(out[j])
            print(f"The parsed output is {outputs}")
            a = [one_option[3:] for one_option in outputs]
            layer_options.extend(a)

        chosen_option = ranking(layer_options, question, status)
        if ("None") in status:
            status = [chosen_option]
        else:
            status.append(chosen_option)
        print(f"The option chosen as the best choice is {chosen_option}")
        print("\n\n\n")

    question_summary = generate_text(summary_question_prompt + str(question), 1)
    predict_prompt_full = (
        predict_prompt
        + str(question_summary)
        + str("Based on the current status - ")
        + str(status)
        + str("\n Just give the answer in number nothing else no text")
    )

    answer = generate_text(predict_prompt_full, 1)

    pred.append(answer[0])
    true.append(true_answer)


    try:
        if float(answer[0]) == true_answer:
            correct += 1
        else:
            wrong += 1
        total += 1
    except:
        wrong += 1
        total += 1
        continue

    questions_big.append(question)
    status_big.append(status)
    print(
        f"Current status is -----------------> correct = {correct} and wrong = {wrong}"
    )
    print(f"Total Tokens: {global_total_tokens}")
    print(f"Prompt Tokens: {global_prompt_tokens}")
    print(f"Completion Tokens: {global_completion_tokens}")
    with open("openai.logs", "a") as log_file:
        log_file.write(
            "\n" + "-----------" + "\n" +  f"Current status is -----------------> correct = {correct} and wrong = {wrong}"+ "\n" + f"Total Tokens: {global_total_tokens}\n" + f"Prompt Tokens: {global_prompt_tokens}\n" + f"Completion Tokens: {global_completion_tokens}\n"
        )
