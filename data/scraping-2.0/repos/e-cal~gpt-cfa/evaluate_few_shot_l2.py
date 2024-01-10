import argparse
import json
import os
import time

import openai
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------------------------
#                                 Parseargs
# ------------------------------------------------------------------------------
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "-f",
    "--file",
    type=str,
    required=True,
    help="path to json exam file with questions and answers",
)
argparser.add_argument(
    "-m",
    "--model",
    type=str,
    default="gpt-3.5-turbo-16k",
    help="gpt model to use (gpt-3.5-turbo or gpt-4)",
)
argparser.add_argument(
    "-t",
    "--temp",
    type=float,
    default=0.0,
    help="temperature to use for gpt response (default 0.0)",
)
argparser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="path to output the attempt's csv file",
)
argparser.add_argument(
    "-c",
    "--chain_of_thought",
    action="store_true",
    help="enable chain-of-thought prompting",
)
argparser.add_argument(
    "-fsr",
    "--few_shot_random",
    action="store_true",
    help="use questions sampled randomly as few shot learning",
)
argparser.add_argument(
    "-fst",
    "--few_shot_topic",
    action="store_true",
    help="use questions sampled from each topic as few shot learning",
)
argparser.add_argument(
    "-n", "--n_shots", type=int, help="number of shots to use for few shot learning"
)
args = argparser.parse_args()
model = args.model
temp = args.temp

if os.path.exists(args.output):
    print(f"output file {args.output} already exists")
    overwrite = input("overwrite? (y/n): ")
    if overwrite.lower() != "y":
        exit(0)

# ------------------------------------------------------------------------------
#                                  Get key
# ------------------------------------------------------------------------------
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
except:
    print("No OpenAI key found")
    exit(1)

# ------------------------------------------------------------------------------
#                               System prompt
# ------------------------------------------------------------------------------
# oneshot_nofunc_prompt = f"""You are a CFA (chartered financial analyst) taking a test to evaluate your knowledge of finance.
# You will be given a question along with three possible answers (A, B, and C).
# Before answering, you should think through the question step-by-step.
# Explain your reasoning at each step towards answering the question.
# If calculation is required, do each step of the calculation as a step in your reasoning.
# Finally, indicate the correct answer (A, B, or C) in double brackets.

# Question:
# Phil Jones, CFA, has just finished researching Alpha One Inc. and is about to issue an unfavorable report on the company. His manager does not want him to state any adverse opinions about Alpha One, as it could adversely affect their firm’s relations with the company, which is an important investment banking client. Which of the following actions by the manager most likely violates Standard I (B): Independence and Objectivity?

# A. Putting Alpha One on a restricted list
# B. Asking Jones to issue a favorable report
# C. Asking Jones to only state facts about the company

# Thinking:
#     - The CFA Institute's Standard I (B): Independence and Objectivity states that a CFA charterholder or candidate must use reasonable care and judgment to achieve and maintain independence and objectivity in their professional activities. They must not offer, solicit, or accept any gift, benefit, compensation, or consideration that reasonably could be expected to compromise their own or another’s independence and objectivity.
#     - In this case, the manager is trying to influence Phil's research report on Alpha One Inc. due to the company's relationship with their firm. This is a clear attempt to compromise Phil's independence and objectivity in his professional activities.
#     - Therefore, the manager's action of trying to prevent Phil from issuing an unfavorable report on Alpha One Inc. most likely violates Standard I (B): Independence and Objectivity.

# [[B]]"""


thinking_prompt = ""
if args.chain_of_thought:
    thinking_prompt = """
    Before answering, you should think through the question step-by-step.
    Explain your reasoning at each step towards answering the question.
    If calculation is required, do each step of the calculation as a step in your reasoning.
    """

func_prompt = f"""You are a CFA (chartered financial analyst) taking a test to evaluate your knowledge of finance.
You will be given a question along with three possible answers (A, B, and C).
{thinking_prompt}
Indicate the correct answer (A, B, or C)."""

sys_prompt = func_prompt

answer_func = {
    "name": "answer_question",
    "description": "Answer a multiple choice question on finance",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the question",
                "enum": ["A", "B", "C"],
            },
        },
        "required": ["answer"],
    },
}

if args.chain_of_thought:
    answer_func["description"] = "Think through and " + answer_func["description"]
    answer_func["parameters"]["required"].append("thinking")
    answer_func["parameters"]["properties"]["thinking"] = {
        "type": "array",
        "items": {
            "type": "string",
            "description": "Thought and/or calculation for a step in the process of answering the question",
        },
        "description": "Step by step thought process and calculations towards answering the question",
    }

if args.few_shot_random or args.few_shot_topic:
    if args.few_shot_random:
        sampling_type = "fsr"
    else:
        sampling_type = "fst"
    if args.chain_of_thought:
        if "level_1" in args.file:
            file_path = f"prompts/l1/{sampling_type}_cot_{args.n_shots}_shot_prompts.json"
        else:
            file_path = f"prompts/l2/{sampling_type}_cot_{args.n_shots}_shot_prompts.json"
    else:
        if "level_1" in args.file:
            file_path = f"prompts/l1/{sampling_type}_{args.n_shots}_shot_prompts.json"
        else:
            file_path = f"prompts/l2/{sampling_type}_{args.n_shots}_shot_prompts.json"
    with open(file_path, "r") as json_file:
        few_shot_prompts = json.load(json_file)

print(f"Few shot prompts: {len(few_shot_prompts)}")


def ask_gpt(question):
    out = None
    function_response = None
    for _ in range(5):
        try:
            messages = [
                {
                    "role": "system",
                    "content": sys_prompt,
                }
            ]
            messages.extend(few_shot_prompts)
            messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )
            res = openai.ChatCompletion.create(
                model=model,
                temperature=temp,
                messages=messages,
                functions=[answer_func],
                function_call={"name": "answer_question"},
            )
            ans = res.choices[0].message.to_dict()["function_call"]["arguments"]  # type: ignore
            # ans = ans.replace("\\", "")
            # ans = ans.replace(u'\u2013', u'')
            # ans = ans.replace(u'U+2013', u'')
            answer = ans.split("'answer': ")[-1].split(",")[0].strip()
            thinking = ans.split("'thinking': ")[-1].strip()
            if args.chain_of_thought:
                answer = answer[1:-1]
                thinking = thinking[1:-2]
            else:
                answer = answer[1:-2]
                thinking = ""
            out = {"answer": answer, "thinking": thinking}
            return out
        except Exception as e:
            print(f"Failed request: {e}")
            time.sleep(5)
            continue

    return {"thinking": "", "answer": "N"}


exam = pd.read_json(args.file)
answers = pd.DataFrame(
    columns=[
        "case",
        "question",
        "chapter_name",
        "choice_a",
        "choice_b",
        "choice_c",
        "answer",
        "guess",
        "correct",
    ]
)

correct = 0
pbar = tqdm(exam.iterrows(), total=len(exam))

i: int
for i, row in pbar:  # type: ignore
    for rowq in row["cfa2_cbt_questions"]:
        question = f"""Case: 
{row["case"]}
Question:
{rowq["question"]}
A. {rowq["choice_a"]}
B. {rowq["choice_b"]}
C. {rowq["choice_c"]}"""

        row_ans = {
            "case": row["case"],
            "question": rowq["question"],
            "chapter_name": rowq["chapter_name"],
            "choice_a": rowq["choice_a"],
            "choice_b": rowq["choice_b"],
            "choice_c": rowq["choice_c"],
            "answer": rowq["answer"]
        }

        gpt_ans = ask_gpt(question)
        row_ans["guess"] = gpt_ans["answer"]

        if gpt_ans["answer"].lower() == rowq["answer"][-1]:
            correct += 1
            row_ans["correct"] = "yes"
        else:
            row_ans["correct"] = "no"

        answers = pd.concat([answers, pd.DataFrame([row_ans])], ignore_index=True)

        pbar.set_postfix({"score": f"{correct}/{i+1} {correct/(i+1) * 100:.2f}%"})

print(f"Score: {correct}/{len(answers)} {correct/len(answers) * 100}%")

print(f"{len(answers[answers['guess'] == 'N'])} failed requests")

answers.to_csv(args.output, index=False)
