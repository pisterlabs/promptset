"""
Data from https://github.com/mrqa/MRQA-Shared-Task-2019
"""

import os
import re
import json
import random
import argparse
import spacy
nlp = spacy.load("en_core_web_sm")

from tqdm import tqdm, trange
from transformers import pipeline, AutoTokenizer
from llm_utils import OpenAI

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='NaturalQuestions', choices=['TriviaQA', 'SQuAD', 'NaturalQuestions', 'NewsQA', 'SearchQA', 'HotpotQA']) #
    parser.add_argument('--n_samples', type=int, default=500) #

    # requirements
    parser.add_argument('--min_ctx_sents', type=int, default=3) #
    parser.add_argument('--min_ctx_tokens', type=int, default=100) #
    parser.add_argument('--max_ctx_tokens', type=int, default=1024) #

    parser.add_argument("--qg_prompt_path", type=str, default="./prompts/qa_qg.txt")
    parser.add_argument("--augment", action="store_true", help="whether to generate relevant questions.")
    parser.add_argument("--verbose", action="store_true", help="whether to print out.")
    parser.add_argument("--debug", action="store_true", help="whether to print out.")

    args, unknown = parser.parse_known_args()


    """
    Step 1: sample data
    """
    # the whole set
    dev_data_path = f"./data/qa/{args.dataset}/{args.dataset}-dev.jsonl"
    dev_data = []
    with open(dev_data_path, 'r') as file:
        for idx, line in enumerate(file):
            # Parse the JSON data in each line
            data = json.loads(line)
            # exclude first line 
            if idx == 0:
                continue 
            dev_data.append(data)
    print("Original data: ", len(dev_data))    
    
    # sample a subset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    sampled_dev_data_path = f"./data/qa/{args.dataset}/dev-{args.n_samples}.json"
    if os.path.exists(sampled_dev_data_path):
        with open(sampled_dev_data_path, "r") as file:
            sampled_dev_data = json.load(file)
    else:
        sampled_dev_data = []

    # record selected samples
    unique_context = []
    for d in sampled_dev_data:
        if d["context"] not in unique_context:
            unique_context.append(d["context"])
    
    # start sampling
    if len(sampled_dev_data) < args.n_samples:
        random.shuffle(dev_data)
        for data in dev_data:
            context = data['context']

            if context in unique_context:
                continue

            # select those with only text, no tables, list
            if any([t in context for t in ['</Table>', '</Tr>', '</Li>', '</Dd>']]):
                continue

            ctx_sents = [str(s) for s in nlp(context.replace("<P>","").replace("</P>","")).sents]
            if len(ctx_sents) < args.min_ctx_sents:
                continue
            
            ctx_len = len(tokenizer(context)["input_ids"])

            if ctx_len < args.min_ctx_tokens:
                continue

            if ctx_len > args.max_ctx_tokens:
                continue
            
            sampled_dev_data.append(data)
            unique_context.append(context)

            if len(sampled_dev_data) == args.n_samples:
                break

    assert len(sampled_dev_data) == args.n_samples
    print("Sampled data: ", len(sampled_dev_data))

    # save dev data
    with open(sampled_dev_data_path, 'w', encoding='utf-8') as file:
        json.dump(sampled_dev_data, file, ensure_ascii=False)


    """
    Step 2: generate relevant questions/answers for dev set
    """
    if args.augment:
        
        with open(sampled_dev_data_path, "r") as file:
            sampled_dev_data = json.load(file)

        augmented_data_path = f"./data/qa/{args.dataset}/dev-{args.n_samples}-aug.json" 
        # load the task data with relevant tasks
        if os.path.exists(augmented_data_path):
            with open(augmented_data_path, "r") as f:
                augmented_dev_data = json.load(f)
        else:
            augmented_dev_data = []
        num_examples = len(augmented_dev_data)
        print(f"Number of existing relevant questions: {num_examples}")

        # load question generation model
        chatgpt = OpenAI(model='gpt-4-1106-preview')
        with open(args.qg_prompt_path, "r") as file:
            qg_prompt = file.read().strip()

        # continue generation
        for idx in trange(num_examples, len(sampled_dev_data)):
            data = sampled_dev_data[idx]

            if len(data['qas']) <= 1: # use gpt-4 to generate

                context = data['context']
                qas = data['qas'][0]
                question = qas['question']
                qid = qas['qid']
                answers = qas['answers']

                prompt = qg_prompt.replace("[[PARAGRAPH]]", context).replace("[[QUESTION]]", question).replace("[[ANSWER]]", answers[0])
                # if args.verbose: print(prompt)
                output = chatgpt.generate(prompt=prompt,
                                            temperature=0.7, 
                                            top_p=1.0, 
                                            max_tokens=128, 
                                            n=1, 
                                            frequency_penalty=0, 
                                            presence_penalty=0, 
                                            stop=["Example", "Question 3"])[0]
                if args.verbose: print(output)

                additional_qa = [x.strip() for x in re.split(r"Question \d+:", output) if x.strip()]
                additional_q = [re.split(r"Answer \d+:", x)[0].strip() for x in additional_qa]
                additional_a = [re.split(r"Answer \d+:", x)[1].strip() for x in additional_qa]

                # append
                for (q, a) in zip(additional_q, additional_a):
                    data['qas'].append(
                        {
                            "question": q,
                            "answers": [a],
                            "id": ""
                        }
                    )

                if args.debug:
                    _ = input("continue...")

            else:
                pass

            augmented_dev_data.append(data)
            with open(augmented_data_path, "w", encoding='utf-8') as file:
                json.dump(augmented_dev_data, file, ensure_ascii=False)