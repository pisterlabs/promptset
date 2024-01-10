import os
import json
import random
from args import get_args
from utils import print_args
import openai
import backoff 
from parse_dataset import parse

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def generate_rationales(args, questions, rationales):
    if len(questions) == 0:
        return None
    else:
        if rationales[0] is None: # no rationales provided
            rationales = []
            for question in questions:
                prompt = "Below is a question provided with the answer. Write intermediate reasoning steps to explain how you get the answer. Split each step by new line.\n" + question
                messages = [{"role": "user", "content": prompt}]
                res = completions_with_backoff(model="gpt-4", 
                                               messages=messages,
                                               temperature=args.temperature,
                                               max_tokens=args.max_length,
                                               )
                rationales.append(res["choices"][0]["message"]["content"])
            return rationales
        else:
            return rationales
        
def main():
    args = get_args()
    print_args(args)

    if args.num_out_domain == 0:
        args.processed_data_dir = os.path.join(args.processed_data_dir,
                                               "in-domain",
                                                (f"i{args.num_in_domain}-s{args.seed}-r{args.rationales}"))
    elif args.num_in_domain == 0:
        args.processed_data_dir = os.path.join(args.processed_data_dir,
                                                "out-domain",
                                                (f"o{args.num_out_domain}-t{args.out_domain_data_name}-s{args.seed}-r{args.rationales}"))
    else:
        args.processed_data_dir = os.path.join(args.processed_data_dir,
                                               "mixed-domain",
                                               (f"i{args.num_in_domain}-o{args.num_out_domain}-t{args.out_domain_data_name}-s{args.seed}-r{args.rationales}"))
    
    if os.path.exists(os.path.join(args.processed_data_dir, "gsm8k.jsonl")):
        print("File already exists, exiting...")
        exit()

    os.makedirs(args.processed_data_dir)

    # load gsm8k data
    with open(os.path.join(args.data_dir, "train.jsonl"), "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    
    with open(os.path.join(args.out_domain_data_dir, "train.jsonl"), "r") as f:
        out_domain_data = [json.loads(line) for line in f.readlines()]

    # template = (
    #     "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    #     "### Instruction:{instruction}\n\n### Demonstration:\n{demonstration}\n\n### Input:{input}\n\n### Response:"
    # )
    template = (
        "### Instruction:{instruction}\n\n{demonstration}\n\nInput:{input}\nOutput:"
    )

    instruction = "Answer the following grade math question."
    
    json_file = open(os.path.join(args.processed_data_dir, f"{args.data_name}.jsonl"), "w")

    random.seed(args.seed)
    indomain_examples = random.sample(data, args.num_in_domain)
    random.seed(args.seed)
    outdomain_examples = random.sample(out_domain_data, args.num_out_domain)
    # exclude indomain_examples from data
    data = [d for d in data if d not in indomain_examples] if args.num_in_domain > 0 else data

    indomain_questions_answer_pair, indomain_questions, indomain_answers, indomain_rationales = [], [], [], []
    outdomain_questions_answer_pair, outdomain_questions, outdomain_answers, outdomain_rationales = [], [], [], []

    for line in indomain_examples:
        question, gold_answer, question_with_answer, rationales = parse(args.data_name, line)
        indomain_questions_answer_pair.append(question_with_answer)
        indomain_questions.append(question)
        indomain_answers.append(gold_answer)
        indomain_rationales.append(rationales)
    for line in outdomain_examples:
        question, gold_answer, question_with_answer, rationales = parse(args.out_domain_data_name, line)
        outdomain_questions_answer_pair.append(question_with_answer)
        outdomain_questions.append(question)
        outdomain_answers.append(gold_answer)
        outdomain_rationales.append(rationales)
    

    indomain_demonstrations, outdomain_demonstrations = "", ""
    if args.rationales:
        indomain_rationales = generate_rationales(args, indomain_questions_answer_pair, indomain_rationales)
        outdomain_rationales = generate_rationales(args, outdomain_questions_answer_pair, outdomain_rationales)
        if indomain_rationales:
            indomain_demonstrations = "\n\n".join([f"Input: {q.strip()}\nOutput: {r.strip()}\nSo the final answer is {a.strip()}" for q, r, a in zip(indomain_questions, indomain_rationales, indomain_answers)])
        if outdomain_rationales:
            outdomain_demonstrations = "\n\n".join([f"Input: {q.strip()}\nOutput: {r.strip()}\nSo the final answer is {a.strip()}" for q, r, a in zip(outdomain_questions, outdomain_rationales, outdomain_answers)])
    else:
        indomain_demonstrations = "\n\n".join([f"Input: {q.strip()}\nOutput: {a.strip()}" for q, a in zip(indomain_questions, indomain_answers)])
        outdomain_demonstrations = "\n\n".join([f"Input: {q.strip()}\nOutput: {a.strip()}" for q, a in zip(outdomain_questions, outdomain_answers)])

    if indomain_demonstrations == "":
        demonstration = outdomain_demonstrations
    elif outdomain_demonstrations == "":
        demonstration = indomain_demonstrations
    else:
        demonstration = indomain_demonstrations + "\n\n" + outdomain_demonstrations

    for line in data:
        question, gold_answer, _, _ = parse(args.data_name, line)
        json_file.write(json.dumps({
            "prompt": template.format(instruction=instruction, 
                                      demonstration=demonstration,
                                      input=question),
            "output": gold_answer,
        }) + "\n")
    
    json_file.close()
    print("Data num", len(data))

if __name__ == '__main__':
    main()