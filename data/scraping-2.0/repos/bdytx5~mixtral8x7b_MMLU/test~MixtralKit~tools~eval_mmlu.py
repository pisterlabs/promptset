import argparse
import openai
import os
import numpy as np
import pandas as pd
import time
import sys
import wandb 
from mixtralkit.mixtral import Mixtral


from crop import crop

openai.api_key = "INSERTYOURKEYHERE"
choices = ["A", "B", "C", "D"]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."



def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval(args, subject, engine, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        while True:
            try:
                c = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=100,
                    temperature=0,
                    echo=True
                )
                break
            except:
                print("pausing")
                time.sleep(1)
                continue

        lprobs = []
        for ans in answers:
            try:
                lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
            except:
                print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
                lprobs.append(-100)
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
# def eval(args, subject, engine, dev_df, test_df):
def eval_hf(args, subject, engine, dev_df, test_df):
    cors = []
    all_probs = []
    answers = ['A', 'B', 'C', 'D']

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True).cuda()

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        
        train_prompt = gen_prompt(dev_df, subject, k) # k samples, (in context learning)

        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)

        out_logits = output.logits[0, -1, :]
        lprobs = []
        for ans in answers:
            token_id = tokenizer.encode(' ' + ans, add_special_tokens=False)[0]
            log_prob = out_logits[token_id].item()
            lprobs.append(log_prob)

        pred = answers[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print(f"Average accuracy {acc:.3f} - {subject}")

    return cors, acc, all_probs

import numpy as np
import torch

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# def eval_mixtral(args, subject, dev_df, test_df):
#     cors = []
#     all_probs = []
#     answers = ['A', 'B', 'C', 'D']

#     max_batch_size = 4
#     max_seq_len = 512
#     max_gen_len = 1
#     temperature = 1.0  # for greedy decoding
#     top_p = 0.9

#     generator = Mixtral.build(
#         ckpt_dir=args.model_weights,
#         tokenizer_path=args.tokenizer,
#         max_seq_len=max_seq_len,
#         max_batch_size=max_batch_size,
#         num_gpus=args.num_gpus,
#     )

#     for i in range(test_df.shape[0]):
#         prompt_end = format_example(test_df, i, include_answer=False)
#         train_prompt = gen_prompt(dev_df, subject, args.ntrain)
#         prompt = train_prompt + prompt_end

#         while crop(prompt) != prompt:
#             args.ntrain -= 1
#             train_prompt = gen_prompt(dev_df, subject, args.ntrain)
#             prompt = train_prompt + prompt_end

#         label = test_df.iloc[i, test_df.shape[1]-1]

#         results = generator.text_completion(
#            [prompt],
#             max_gen_len=max_gen_len,
#             temperature=temperature,
#             top_p=top_p,
#         )

#         rpred, pred = results[0]['generation'], results[0]['generation'][0]
#         cor = pred == label
#         print("###",label, pred, "\n")
        
#         cors.append(cor)
#         # all_probs.append(probs)

#     acc = np.mean(cors)
#     cors = np.array(cors)
#     all_probs = np.array([0.,0.,0.,1.])
#     print(f"Average accuracy {acc:.3f} - {subject}")

#     return cors, acc, all_probs


def eval_mixtral(args, subject, dev_df, test_df, generator):
    cors = []
    all_probs = []
    answers = ['A', 'B', 'C', 'D']

    max_gen_len = 1
    temperature = 0.0  # for greedy decoding
    top_p = 0.9

    # Initialize wandb.Table for logging
    results_table = wandb.Table(columns=["Question", "Predicted Answer", "Correct Answer", "Correct"])

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, args.ntrain)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            args.ntrain -= 1
            train_prompt = gen_prompt(dev_df, subject, args.ntrain)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        results = generator.text_completion(
            [prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        try:
            rpred, pred = results[0]['generation'], results[0]['generation'][0]
            cor = pred == label
            print("###",label, pred, "\n")
            
            cors.append(cor)
            results_table.add_data(prompt, pred, label, cor)
        except:
            results_table.add_data(prompt, "error", label, False)
            continue

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array([0.,0.,0.,1.])
    print(f"Average accuracy {acc:.3f} - {subject}")

    # Log the table to wandb
    wandb.log({f"{subject}_results_table": results_table})

    return cors, acc, all_probs

# def eval_mixtral(args, subject, dev_df, test_df, generator):
#     cors = []
#     all_probs = []
#     answers = ['A', 'B', 'C', 'D']

#     max_gen_len = 1
#     temperature = 1.0  # for greedy decoding
#     top_p = 0.9

#     for i in range(test_df.shape[0]):
#         prompt_end = format_example(test_df, i, include_answer=False)
#         train_prompt = gen_prompt(dev_df, subject, args.ntrain)
#         prompt = train_prompt + prompt_end

#         while crop(prompt) != prompt:
#             args.ntrain -= 1
#             train_prompt = gen_prompt(dev_df, subject, args.ntrain)
#             prompt = train_prompt + prompt_end

#         label = test_df.iloc[i, test_df.shape[1]-1]

#         results = generator.text_completion(
#             [prompt],
#             max_gen_len=max_gen_len,
#             temperature=temperature,
#             top_p=top_p,
#         )

#         rpred, pred = results[0]['generation'], results[0]['generation'][0]
#         cor = pred == label
#         print("###",label, pred, "\n")
        
#         cors.append(cor)

#     acc = np.mean(cors)
#     cors = np.array(cors)
#     all_probs = np.array([0.,0.,0.,1.])
#     print(f"Average accuracy {acc:.3f} - {subject}")

#     return cors, acc, all_probs

def main(args):
    engines = args.engine
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
            os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))
    
    
    wandb.init(project="mixtral_evaluation", name="Mixtral Evaluation Run MMLU")

    print(subjects)
    print(args)

    max_batch_size = 4
    max_seq_len = 2048*2

    generator = Mixtral.build(
        ckpt_dir=args.model_weights,
        tokenizer_path=args.tokenizer,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        num_gpus=args.num_gpus,
    )
    
    for engine in engines:
        print(engine)
        all_cors = []

        for subject in subjects:
            try:
                dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
                test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

                cors, acc, probs = eval_mixtral(args, subject, dev_df, test_df, generator=generator)
                wandb.log({f"{subject}_accuracy": acc})
                all_cors.append(cors)
                
                test_df["{}_correct".format(engine)] = cors
   

            except:
                print("error for subject {}".format(subject))
                continue

        weighted_acc = np.mean(np.concatenate(all_cors))
        print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    

    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/root/Users/brettyoung/Desktop/mixtral/test/data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--engine", "-e", choices=["mixtral"],
                        default="mixtral", nargs="+")

    # Adding new arguments for Mixtral model
    parser.add_argument("--model_weights", "-m", type=str, required=True,
                        help="Path to the Mixtral model weights.")
    parser.add_argument("--tokenizer", "-t", type=str, required=True,
                        help="Path to the tokenizer file.")
    parser.add_argument("--num_gpus", "-g", type=int, default=8,
                        help="Number of GPUs to use.")

    args = parser.parse_args()
    main(args)


