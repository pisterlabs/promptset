import os
import argparse
import openai
import json
from tqdm import tqdm as tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_type", type=str, default="cot") # choose between cot or subques
    parser.add_argument("--resample_path", type=str, default="./results/cot_sample_subques_resample.jsonl")
    parser.add_argument("--result_path", type=str, default="./results/cot_sample_subques_resample_selection.jsonl")
    parser.add_argument("--prompt_path", type=str, default="./prompts/cot_selection.txt")
    parser.add_argument("--hetero", type=bool, default=False)
    parser.add_argument("--openai_key", type=str, default="")

    return parser.parse_args()

#read prompt file
def read_prompt(PROMPT_PATH):
    with open(PROMPT_PATH, "r") as f:
        prompt = f.read()
    return prompt

# read data from the file
def get_dataset(PATH):
    with open(PATH, 'r') as dataset:
        data_list = list(dataset)
        for line in data_list:
            problem = json.loads(line)
            yield problem

# save results to a file
def store_result(out, problem, output):
    problem['selection'] = output
    out.write(json.dumps(problem, ensure_ascii=False) + '\n')


def selection_cot(SAMP_PATH, RES_PATH, PROMPT_PATH, HETERO):

    BASE_PROMPT = read_prompt(PROMPT_PATH)
    
    with open(RES_PATH, "a") as out:
        for sample in tqdm(get_dataset(SAMP_PATH)):
            question = sample["question"]
            answerA = sample["prediction"]
            if HETERO:
                answerB = sample["het_resample"]
            else:
                answerB = sample["resample"].split("Final Answer: ")[-1]
            instruction = "You are an expert math teacher. You are provided with a question and two answers. Lets check the 'Answer choices' step by step, and then decide which answer is correct '(A)' or '(B)'"
            prompt = f"{BASE_PROMPT}\n{instruction}\nQuestion: {question}\nAnswer choices:\n(A) {answerA}\n(B) {answerB}\nAnswer: ("

            not_responded_yet = True
            while not_responded_yet:
                try:
                    response = openai.Completion.create(
                                model="gpt-3.5-turbo-instruct",
                                prompt=prompt,
                                max_tokens=2,
                                temperature=0
                            )
                    not_responded_yet = False
                except:
                    print("retrying\n")

                output = response.choices[0]["text"]
                if "A" in output:
                    final = answerA
                elif "B" in output:
                    final = answerB
                else:
                    final = output
            store_result(out, sample, final)


def selection_subques(SAMP_PATH, RES_PATH, PROMPT_PATH, HETERO):

    BASE_PROMPT = read_prompt(PROMPT_PATH)
    
    with open(RES_PATH, "a") as out:
        for sample in tqdm(get_dataset(SAMP_PATH)):
            question = sample["question"]
            answerA = " ".join([qa[1].split("The answer is ")[0] for qa in sample["prediction"][:-1]]) + sample["prediction"][-1][1]
            if HETERO:
                answerB = sample["het_resample"]
            else:
                answerB = " ".join([qa[1].split("The answer is ")[0] for qa in sample["resample"][:-1]])+ sample["resample"][-1][1]
            instruction = "You are an expert math teacher. You are provided with a question and two answers. Lets check the 'Answer choices' step by step, and then decide which answer is correct '(A)' or '(B)'"
            prompt = f"{BASE_PROMPT}\n{instruction}\nQuestion: {question}\nAnswer choices:\n(A) {answerA}\n(B) {answerB}\nAnswer: ("

            not_responded_yet = True
            while not_responded_yet:
                try:
                    response = openai.Completion.create(
                                model="gpt-3.5-turbo-instruct",
                                prompt=prompt,
                                max_tokens=2,
                                temperature=0
                            )
                    not_responded_yet = False
                except:
                    print("retrying\n")

            output = response.choices[0]["text"]
            if "A" in output:
                final = sample["prediction"]
            elif "B" in output:
                final = sample["resample"]
            else:
                print ("Neither A nor B", output)
                final = output
            store_result(out, sample, final)

def main(args):
    # call inference function for sampling
    openai.api_key = args.openai_key
    if args.sampling_type == "cot":
        selection_cot(args.resample_path, args.result_path, args.prompt_path, args.hetero)

    elif args.sampling_type == "subques":
        selection_subques(args.resample_path, args.result_path, args.prompt_path, args.hetero)
    else:
        print ("Choose between 'cot' or 'subques'")


if __name__ == "__main__":
    args = get_args()
    assert args.resample_path != "", "Please provide the resampled output path for GSM8K"
    if args.prompt_path == "":
        print ("Sampling in Zero Shot Setting!")

    main(args)