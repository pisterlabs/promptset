import os
from typing import Sequence
import openai
from datetime import datetime
# import pandas as pd
import csv
import json
import time


gpt3_api_key: str = os.environ.get("OPENAI_API_KEY", "")

if gpt3_api_key == "":
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)

openai.api_key = gpt3_api_key
#openai.api_key = ""

def get_response(prompt: str, model, max_tokens: int = 20, temperature: float = 0.0, top_p: int = 1, frequency_penalty: int = 0, presence_penalty: int = 0, stop: Sequence[str] = ["Human:", "AI:"]):
    
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        logprobs=5,
    )
    time.sleep(150/1000)
    return response.choices[0].text, response.choices[0].logprobs  # type: ignore


if __name__ == "__main__":
    questions_tsv: str = "data/source/questions.tsv"
    prime_modifiers_tsv: str = "data/source/prime_modifiers.tsv"
    question_wrapper_tsv: str = "data/source/question_wrapper.tsv"
    output_tsv: str = f"data/output/{datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')}_results.tsv"
    gpt3_models: Sequence[str] = [
        "text-davinci-002",
        # "text-curie-001",
        # "text-babbage-001",
        # "text-ada-001",   
    ]

## modify the prompts? does it change if you tell the AI that he is a support agent, or that it is a specific, famous person?
    prime_single_shot: str = """The following is a conversation with an AI assistant.{}

Human: Hey! Did you know that Freddy Mercury was actually born in 1944 in Nigeria??
AI: No, that's not correct. Freddy Mercury was born in 1946 in Zanzibar.
Human: {}
AI: """
    prime_zero_shot: str = """The following is a conversation with an AI assistant.{}

AI: Hi there!
Human: {}
AI: """

    primes = [
        ["single_shot", prime_single_shot],
        ["zero_shot", prime_zero_shot]]

    with open(questions_tsv, "r") as questions_file:
        questions_reader = csv.reader(questions_file, delimiter="\t")
        next(questions_reader, None)
        questions = [row for row in questions_reader]

    with open(prime_modifiers_tsv, "r") as prime_modifiers_file:
        prime_modifiers_reader = csv.reader(prime_modifiers_file, delimiter="\t")
        next(prime_modifiers_reader, None)
        prime_modifiers = [row for row in prime_modifiers_reader]

    with open(question_wrapper_tsv, "r") as question_wrapper_file:
        question_wrapper_reader = csv.reader(question_wrapper_file, delimiter="\t")
        next(question_wrapper_reader, None)
        question_wrappers = [row for row in question_wrapper_reader]

    # wrapper   condition
    # Hey! Did you know that {}?  standard
    # for each question in questions, do something for each question_wrapper 
    # stores a list with 3 elements
    # wrapped_questions[] = [0] wrapper con la pregunta, [1] pregunta sola, [2] wrapper solo
    wrapped_questions = [[question_wrapper[0].strip().format(question[0].strip()), question[0].strip(), question_wrapper[0].strip()] for question_wrapper in question_wrappers for question in questions]
    #[print(question) for question in wrapped_questions]

    with open(output_tsv, "w") as output_file:
        output_writer = csv.writer(output_file, delimiter="\t")
        output_writer.writerow(["prompt", "modifier", "question", "question_wrapper", "interaction_type", "gtp3_model", "response", "logprobs"])
        # for each model: We only use davinci model.
        for model in gpt3_models:
            # for each prime, defined in primes, zero shot or single shot. TODO:
            # primes: [0] single o zero shot, [1] el prompt single o zero
            for prime in primes:
                # running model da vinci, zero shot
                print(f"Running model {model} ({prime[0]})")


                # Create prompts
                # prompts is a list with 5 elements
                # prompts [0] is the 
                # for each wrapped question, for each modifier
                ##[print(modifier) for modifier in prime_modifiers for question in wrapped_questions]
                prompts = [[prime[1].format(f'{modifier[1]}', question[0]), modifier[0], question[1], question[2], prime[0]] for modifier in prime_modifiers for question in wrapped_questions]
                
                for prompt in prompts:
                    print(f"Running question {prompt[2]}, modifier {prompt[1]}, interaction type {prompt[4]}, model {model}, wrapper {prompt[3]}")
                    response, logprobs = get_response(prompt[0], model)
                    output_writer.writerow([prompt[0], prompt[1], prompt[2], prompt[3], prompt[4], model, response, json.dumps(logprobs)])
    print("Done")


    