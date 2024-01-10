"""
Let Large Language Model fulfill the test.
"""
import json
import backoff
import openai
import argparse
import pickle
import os
from tqdm import tqdm

PROMPT_PATH = "{}_prompt_intj_simple.txt"
QUESTIONS_PATH = '{}_questions.txt'
OPENAI_API_KEY = "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" # TODO: Add your own API.


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIConnectionError))
def call_api(engine: str, prompt: str, max_tokens, temperature):
    return openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=['--', '\n\n', ';', '#'],
    )


if __name__ == '__main__':
    openai.api_key = OPENAI_API_KEY
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default="code-davinci-002-with-intj-simple-claim")
    parser.add_argument('--benchmark', type=str, default="mbti")
    parser.add_argument('--engine', type=str, default="code-davinci-002")
    parser.add_argument('--max_tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)

    args = parser.parse_args()

    with open(PROMPT_PATH.format(args.benchmark), "r") as f:
        raw_prompt = f.read()

    with open(QUESTIONS_PATH.format(args.benchmark), "r") as f:
        questions = [line.strip() for line in f.readlines()]

    prompts = []
    raw_results = []
    result_strs = []
    last_6qa = []
    prompt = "{}\n\n".format(raw_prompt)

    for question in tqdm(questions):
        if len(prompt) < 10000:
            prompt += "Q: {}\nA:".format(question)
        else:
            prompt = "{}\n\n".format(raw_prompt)
            prompt += "\n\n".join(["Q: {}\nA: {}".format(qa_pair[0], qa_pair[1]) for qa_pair in last_6qa[-6:]])
        prompts.append(prompt)
        result = call_api(engine=args.engine, prompt=prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        raw_results.append(result)
        result_str = result['choices'][0]['text'].strip()
        result_strs.append(result_str)
        last_6qa.append((question, result_str))
        prompt += ' {}\n\n'.format(result_str)

    os.makedirs(f"{args.benchmark}/prompts", exist_ok=True)
    os.makedirs(f"{args.benchmark}/results", exist_ok=True)
    os.makedirs(f"{args.benchmark}/result_strs", exist_ok=True)

    with open(f'{args.benchmark}/prompts/prompts_{args.run_id}.pkl', 'wb') as f:
        pickle.dump(prompts, f)

    with open(f'{args.benchmark}/results/results_{args.run_id}.pkl', 'wb') as f:
        pickle.dump(raw_results, f)

    with open(f'{args.benchmark}/result_strs/result_strs_{args.run_id}.json', 'w') as f:
        json.dump(result_strs, f)
