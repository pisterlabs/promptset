import argparse
import sys

import openai
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import random
from base_prompt1 import *
from learn_policy1 import get_gpt3_output
import torch.nn as nn
import torch
import torch.nn.functional as F  
from utilities1 import extract_prediction,normalize_answer
import numpy as np
from learn_policy1 import load_data

class policy_network(nn.Module):

    def __init__(self,
                 model_config="bert-base-uncased",
                 add_linear=False,
                 embedding_size=128,
                 freeze_encoder=True) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        print("model_config:", model_config)
        self.model = AutoModelForTokenClassification.from_pretrained(model_config)

        # Freeze transformer encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(self.model.config.hidden_size,
                                    embedding_size)  # 768 for bert-base-uncased, distilbert-base-uncased
        else:
            self.linear = None

    def forward(self, input_list):
        input = self.tokenizer(input_list, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
        # print(f"input: {input}")
        output = self.model(**input, output_hidden_states=True)
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]
        # Get [CLS] hidden states
        sentence_embedding = last_hidden_states[:, 0, :]  # len(input_list) x hidden_size
        # print(f"sentence_embedding: {sentence_embedding}")

        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding
    



def test_policy_network(args):
    problems,cand_pids,test_pids = load_data(args)

    ctxt_list = [problems[pid] for pid in test_pids]
    cands_list = [problems[pid] for pid in cand_pids]

    model = policy_network(model_config="bert-base-uncased", add_linear=True, embedding_size=256)
    # 捕捉IndexError: list index out of range
    try:
        model.load_state_dict(torch.load("checkpoints/svampexp/model1.pt"))
    except IndexError:
        print("IndexError: list index out of range")
        sys.exit(0)
    embedding_ctxt = model(ctxt_list)
    embedding_cands= model(cands_list)
    scores = torch.mm(embedding_ctxt, embedding_cands.t())  # len(train_batch) x len(cand_examples)
            # print(f"unnormed scores: {scores}")
    scores = F.softmax(scores, dim=1)  # len(train_batch) x len(cand_examples)
    print(f"scores: {scores}")
    scores = scores.cpu().detach().numpy()
    for i, test_pid in enumerate(test_pids):
        print(f"test_problem: {problems[test_pid]}")
        score = scores[i, :].tolist()
        cand_prob = scores[i,:].clone().detach()
        cand_rank = sorted(range(len(score)), key=lambda i: scores[i], reverse=True)
        cand_pids = [cand_pids[cid] for cid in cand_rank]
        print(f"====== candidates rank: {[problems[pid] for pid in cand_pids]}")
        
        # sample shot_pids from the cand_prob distribution
        cids = np.random.choice(range(len(cand_pids)), args.shot_number, p=cand_prob, replace=False)

        # reverse shot_pids so more relevant prompt will be put closer to the question
        cids = cids[::-1]
        # print(f"cids: {cids}")

        shot_pids = [cand_pids[cid] for cid in cids]
        # print(f"shot_pids: {shot_pids}")

        # generate the prompt input
        prompt = build_prompt(problems, shot_pids, test_pids[i], args)

        # get the output from GPT-3
        output = get_gpt3_output(prompt, args)

        # extract the prediction from the output
        prediction = extract_prediction(output)
        print(f"test_pid:{test_pid}")
        print(f"prediction: {prediction}")
        print(f'Answer: {problems[test_pid]["Answer"]}')

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Policy gradient settings
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    # parser.add_argument('--train_number', type=int, default=20, help='Number of training samples.')
    parser.add_argument('--cand_number', type=int, default=10, help='Number of candidate prompts.')
    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002', choices=['text-davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    args = parser.parse_args()

    sys.path.append("../")
    openai.api_key = "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"
    test_policy_network(args)

