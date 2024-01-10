"""Generate answers with GPT-3.5"""
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import json
import os
import time
import concurrent.futures
import openai
import tqdm
import shortuuid
from llmeval.file_io import read_file,write_jsonl
openai.proxy = {"https":"http://127.0.0.1:7890","http":"http://127.0.0.1:7890"}

def get_answer(question_id: int, question: str, max_tokens: int,
               model = "gpt-3.5-turbo",
               model_id= "gpt-3.5-turbo:20230327"):
    ans = {
        "answer_id": shortuuid.uuid(),
        "question_id": question_id,
        "model_id": model_id,
    }
    for _ in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                max_tokens=max_tokens,
            )
            ans["text"] = response["choices"][0]["message"]["content"]
            return ans
        except Exception as e:
            print("[ERROR]", e)
            ans["text"] = "#ERROR#"
            time.sleep(1)
    return ans


def baseline_inference(args):
    # 1 read data
    questions  = read_file(args.input)
    questions_dict = {}
    for idx, ques in enumerate(questions):
        questions_dict[idx] = ques

    # 2 call chatgpt
    answers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for qid, question in questions_dict.items():
            future = executor.submit(get_answer, qid, question, args.max_tokens, args.base_model)
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            answers.append(future.result())

    answers.sort(key=lambda x: x["question_id"])

    # 3 save result
    outfile = os.path.join(args.outdir,"answer_gpt35.jsonl")
    write_jsonl(answers,outfile)

    return answers
