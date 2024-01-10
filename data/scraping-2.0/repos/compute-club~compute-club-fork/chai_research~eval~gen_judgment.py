import argparse
import ast
import json
import os
import re
import time

import openai
import pandas as pd
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

JUDGE = "gpt-4" # or, "gpt-3.5-turbo"
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")


def load_questions(question_file):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_file):
    """Load model answers."""
    answer = {}
    with open(answer_file) as fin:
        for line in fin:
            line = json.loads(line)
            answer[line["question_id"]] = line
    return answer


def load_judge_prompts(prompt_file):
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def chat_compeletion_openai(messages):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=JUDGE,
                messages=messages,
                n=1,
                temperature=0,
                max_tokens=2048
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def run_judge_single(judge_prompt, question, answer):
    rating = -1

    user_prompt = judge_prompt["prompt_template"].format(
        question_1=question["turns"][0],
        question_2=question["turns"][1],
        answer_1=answer["choices"][0]["turns"][0],
        answer_2=answer["choices"][0]["turns"][1]
    )
    
    system_prompt = judge_prompt["system_prompt"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    judgment = chat_compeletion_openai(messages)

    if judge_prompt["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
    else:
        raise ValueError(
            f"invalid output format: {judge_prompt['output_format']}"
        )

    return rating, user_prompt, judgment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=False,
        help="The path to the lora config. This can be a local folder or a Hugging Face repo ID.",
    )
    args = parser.parse_args()

    # Load questions
    questions = load_questions("question.jsonl")

    # Load answers
    if args.lora_path:
        model_name = args.lora_path.split('/')[-1]
    else:
        model_name = args.model_path.split('/')[-1]
    answer_file = model_name + ".jsonl"
    model_answers = load_model_answers(answer_file)

    # Load judge
    judge_prompt = load_judge_prompts("judge_prompts.jsonl")["single-v1-multi-turn"]

    # Generate judgments
    records = []
    for question in tqdm(questions):
        answer = model_answers[question["question_id"]]
        rating, user_prompt, judgment = run_judge_single(
            judge_prompt, question, answer
        )
        question_id = question["question_id"]
        result = {
            "question_id": question_id,
            "model": model_name,
            "judge": JUDGE,
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": rating,
            "tstamp": time.time(),
        }
        print(
            f"question: {question_id}, model: {model_name}, "
            f"score: {rating}, "
            f"judge: {JUDGE} "
        )
        output_file = "judgments.jsonl"
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")
    
    # Average results
    df_all = pd.read_json(output_file, lines=True)
    df = df_all[["model", "score"]].groupby(["model"]).mean()
    df = df.reset_index()
    print("Average results:")
    print(df.sort_values(by="score", ascending=False))
    df.to_json("results.jsonl", orient="records", lines=True)
