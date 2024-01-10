import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, GPTNeoForCausalLM, AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from scipy.special import softmax
from sample_metaphors_multi_hop import multi_hop_metaphors
import pdb
import pandas as pd
import math
from typing import List
import random
import argparse

prompt_file = "./data/common_metaphors.txt"

def model_init(model_string, cuda, output_attentions=False, fast=False):
    if model_string.startswith("gpt2"):
        if fast:
            tokenizer = AutoTokenizer.from_pretrained(model_string)
            model = GPT2LMHeadModel.from_pretrained(model_string)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(model_string) 
            model = GPT2LMHeadModel.from_pretrained(model_string)
    elif model_string.startswith("EleutherAI/gpt-neo"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_string, output_attentions=output_attentions)
        model = GPTNeoForCausalLM.from_pretrained(model_string, output_attentions=output_attentions)
    else:
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_string)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_string)
    model.eval()
    if cuda:
        model.to('cuda')
    return model, tokenizer


def sent_scoring(model_tokenizer, text, cuda, score_type="loss", output_attentions=False, length_normalize=False):
    model = model_tokenizer[0]
    tokenizer = model_tokenizer[1]
    assert model is not None
    assert tokenizer is not None
    encoded_text = tokenizer.encode(text)
    input_ids = torch.tensor(encoded_text).unsqueeze(0) 
    if cuda:
        input_ids = input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, output_attentions=output_attentions)
    loss, logits = outputs[:2]

    sentence_prob = loss.item()
    if score_type == "prob":
        if length_normalize:
            mult = 2
        else:
            mult = len(encoded_text)

        sentence_prob = math.exp(-1.0 * loss * (mult - 1))

    if output_attentions:
        attn = outputs["attentions"]
        return sentence_prob, attn, input_ids
        
    return sentence_prob

def confusion_matrix(P_forward_1, P_forward_2, P_backward_1, P_backward_2):
    correct_forward = len(np.where(np.array(P_forward_1) >= 0.5)[0]) + len(np.where(np.array(P_forward_2) >=0.5)[0])
    wrong_forward = len(P_forward_1) + len(P_forward_2) - correct_forward

    correct_backward = len(np.where(np.array(P_backward_1) >= 0.5)[0]) + len(np.where(np.array(P_backward_2) >=0.5)[0])
    wrong_backward = len(P_backward_1) + len(P_backward_2) - correct_backward

    print("correct forward", correct_forward, "wrong forward", wrong_forward, "correct backward", correct_backward, "wrong_backward", wrong_backward)

def evaluate_model(model, tokenizer, test_set, middle_phrase="", use_prefix=0, verbose=True, score_type="prob", use_cuda=False, return_acc=False) -> tuple:
    preds = []
    labels = []
    x_1 = []
    x_2 = []
    y_1 = []
    y_2 = []
    P_x_1 = []
    P_x_2 = []
    P_y_1 = []
    P_y_2 = []
    P_x_1_y_1 = []
    P_x_1_y_2 = []
    P_x_2_y_1 = []
    P_x_2_y_2 = []
    P_x_1_correct = []
    P_x_2_correct = []
    P_y_1_correct = []
    P_y_2_correct = []
    correct = 0

    for i, metaphor_data in enumerate(test_set):
        ctx, p1, p2 = metaphor_data["startphrase"], metaphor_data["ending1"], metaphor_data["ending2"]
        labels.append(int(metaphor_data["labels"]))
        if use_prefix > 0:
            prefix_prompt = select_prefix_prompts(prompt_file, use_prefix) if use_prefix else ""
        else:
            prefix_prompt = ""

        sent1 = prefix_prompt + ctx + ". " + middle_phrase + p1 + "."
        sent2 = prefix_prompt + ctx + ". " + middle_phrase + p2 + "."

        score1 = sent_scoring((model, tokenizer), sent1, use_cuda, score_type=score_type)
        score2 = sent_scoring((model, tokenizer), sent2, use_cuda, score_type=score_type)

        if score_type == "loss":
            pred = 0 if score1 < score2 else 1
        else:
            pred = 1 if score1 < score2 else 0

        pred_sent = sent1 if pred == 0 else sent2

        if i % 2 == 0:
            x_1.append(ctx)
            x_1_score = sent_scoring((model, tokenizer), ctx + ".", use_cuda, score_type=score_type)
            P_x_1.append(x_1_score)
            y_1.append(p1)
            y_2.append(p2)
            y1_score = sent_scoring((model, tokenizer), p1 + ".", use_cuda, score_type=score_type)
            y2_score = sent_scoring((model, tokenizer), p2 + ".", use_cuda, score_type=score_type)
            P_y_1.append(y1_score)
            P_y_2.append(y2_score)

            P_x_1_y_1.append(score1)
            P_x_1_y_2.append(score2)
            P_x_1_correct.append(score1/(score1 + score2))

        else:
            x_2.append(ctx)
            x_2_score = sent_scoring((model, tokenizer), ctx + ".", use_cuda, score_type=score_type)
            P_x_2.append(x_2_score)
            P_x_2_y_1.append(score1)
            P_x_2_y_2.append(score2)
            P_x_2_correct.append(score2/(score1 + score2))

            P_y_1_correct.append(P_x_1_y_1[-1]/(P_x_1_y_1[-1] + score1))
            P_y_2_correct.append(score2/(P_x_1_y_2[-1] + score2))
        
        if verbose:
            print(f"Q: {ctx}: 1. {p1} 2. {p2}")
            print(f"model says '{pred_sent}' is more likely")
            print("\n")
        if pred == metaphor_data["labels"]:
            correct += 1
        preds.append(pred)

    cols = {"x_1": x_1, "x_2": x_2, "y_1": y_1, "y_2": y_2, "P(x_1)": P_x_1, "P(x_2)": P_x_2, "P(y_1)": P_y_1, "P(y_2)": P_y_2,
        "P(x_1, y_1)": P_x_1_y_1, "P(x_1, y_2)": P_x_1_y_2, "P(x_2, y_1)": P_x_2_y_1, "P(x_2, y_2)": P_x_2_y_2,
        "P(y_1|x_1)": P_x_1_correct, "P(y_2|x_2)": P_x_2_correct, "P(x_1|y_1)": P_y_1_correct, "P(x_2|y_2)": P_y_2_correct}
    out_df = pd.DataFrame(cols)

    if return_acc:
        return correct/len(preds), out_df, preds, labels

    return out_df, preds, labels
 
def evaluate_model_multi_hop(model, tokenizer, test_set, score_type="prob", use_cuda=False, return_acc=False, keep_errors=False) -> tuple:
    pair_correctness = {}
    first_correct = []
    second_correct = []
    all_correct = 0
    labels = []
    preds = []
    prev_answer = None
    prev_correct_answer = None

    for i, metaphor_data in enumerate(test_set):
        ctx, p1, p2, label, qid, hop_num = metaphor_data["startphrase"], metaphor_data["ending1"], metaphor_data["ending2"], metaphor_data["labels"], metaphor_data["qid"], metaphor_data["hop"]
        correct_answer = p1 if label == 0 else p2

        labels.append(int(metaphor_data["labels"]))
        if hop_num == 1:
            ctx = ctx.replace("[ANS]", prev_answer) if keep_errors else ctx.replace("[ANS]", prev_correct_answer)

        sent1 = ctx + middle_phrase + " " + p1 
        sent2 =  ctx + middle_phrase + " " + p2 
        score1 = sent_scoring((model, tokenizer), sent1, use_cuda, score_type=score_type)
        score2 = sent_scoring((model, tokenizer), sent2, use_cuda, score_type=score_type)

        if score_type == "loss":
            pred = 0 if score1 < score2 else 1
        else:
            pred = 1 if score1 < score2 else 0

        pred_sent = sent1 if pred == 0 else sent2
        if hop_num == 0:
            prev_answer = pred_sent
            prev_correct_answer = ctx + middle_phrase + " " + correct_answer

        preds.append(pred_sent)
        labels.append(label)
        
        if qid not in pair_correctness:
            pair_correctness[qid] = {hop_num: pred == label} 
        else:
            pair_correctness[qid][hop_num] = pred == label

        if hop_num == 0 and pred == label:
            first_correct.append(qid)
        elif hop_num == 1 and pred == label: 
            second_correct.append(qid)
            if len(first_correct) > 0 and first_correct[-1] == qid:
                all_correct += 1


    first_correct = len(first_correct)/len(pair_correctness)
    second_correct = len(second_correct)/len(pair_correctness)
    all_correct = all_correct / len(pair_correctness)

    return first_correct, second_correct, all_correct

def select_prefix_prompts(prompt_filepath: str, num_prompts: int = 3) -> None:
    with open(prompt_filepath, "r") as f:
        lines = f.readlines()

    return " ".join(random.sample(lines, num_prompts))

def compute_stats(total_df: pd.DataFrame, all_preds: List, all_labels: List) -> None:
    print("overall accuracy: ")
    print(len(np.where(np.array(all_preds) == np.array(all_labels))[0])/len(all_labels))
    print("confusion matrix: ")
    confusion_matrix(list(total_df["P(y_1|x_1)"]), list(total_df["P(y_2|x_2)"]), list(total_df["P(x_1|y_1)"]), list(total_df["P(x_2|y_2)"]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models with the causal language modelling objective (GPT-*)")
    parser.add_argument("model_id", choices=["gpt2", "gpt-neo-sm", "gpt-neo-lg"]) 
    parser.add_argument("--middle_phrase", default="")
    parser.add_argument("--score_type", default="prob", choices=["prob", "loss"])
    parser.add_argument("--use_prefix", type=int, choices=range(1, 21), default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--multi-hop", action="store_true")
    parser.add_argument("--out_file")
    args = parser.parse_args()
    
    use_cuda = args.cuda
    model_names = {"gpt2": "gpt2", "gpt-neo-sm": "EleutherAI/gpt-neo-1.3B", "gpt-neo-lg": "EleutherAI/gpt-neo-2.7B"}
    model_id = args.model_id
    model_name = model_names[model_id]
    if args.out_file:
        tsv_name = args.out_file
    else:
        tsv_name = f"{model_id}_prob" if not args.use_prefix else f"{model_id}_prob_prefix"
    middle_phrase = args.middle_phrase
    score_type = args.score_type
    verbose = args.verbose

    if args.multi_hop:
        model, tokenizer = model_init(model_name, use_cuda)
        metaphor_set = multi_hop_metaphors["test"]
        first_correct, second_correct, all_correct = evaluate_model_multi_hop(model, tokenizer, metaphor_set)
        print("first hop: ", first_correct)
        print("second hop: ", second_correct)
        print("both correct: ", all_correct)
    else:
        model, tokenizer = model_init(model_name, use_cuda)
        metaphor_data = pd.read_csv("./data/filtered/dev.csv")

        total_df, all_preds, all_labels = evaluate_model(model, tokenizer, metaphor_data.to_dict(orient="records"), use_cuda=use_cuda, verbose=verbose, middle_phrase=middle_phrase, use_prefix=args.use_prefix)
        
        compute_stats(total_df, all_preds, all_labels)
        total_df.to_csv(f"{tsv_name}.tsv", sep="\t", index=False)
