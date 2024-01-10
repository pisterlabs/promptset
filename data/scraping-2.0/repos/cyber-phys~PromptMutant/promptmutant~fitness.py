import os
import openai
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
from datasets import load_dataset
import torch
import logging
import random
import re
import sys

def bert_encode(texts):
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def cosine_similarity_score(prompt, training_set, llm):
    seed = random.randint(0, 1000000)
    shuffled_set = training_set.shuffle(seed=seed)
    question_set = shuffled_set["question"][:5]
    answer_set = shuffled_set["answer"][:5]

    total_similarity = 0
    for i, question in enumerate(question_set):
        response = llm(prompt + "\n" + question)
        response_embedding = bert_encode([response])
        answer_embedding = bert_encode([answer_set[i]])
        similarity = cosine_similarity(response_embedding, answer_embedding)
        total_similarity += similarity[0][0]
        
    average_similarity = total_similarity / len(question_set)
    return average_similarity

## Code forked from https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def check_last_line_for_number(text, x):
    last_line = text.strip().split('\n')[-1]
    return str(x) in last_line

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example)
    assert gt_answer != INVALID_ANS
    # return extract_answer(model_completion) == gt_answer
    return check_last_line_for_number(model_completion, gt_answer)

def gsm8k_score(prompt, training_set, llm):
    seed = random.randint(0, 1000000)
    shuffled_set = training_set.shuffle(seed=seed)
    question_set = shuffled_set["question"][:5]
    answer_set = shuffled_set["answer"][:5]
    score = 0
    for i, question in enumerate(question_set):
        response = llm(prompt + "\n" + question)
        if is_correct(response, answer_set[i]):
            score += 1
            sys.stdout.write("✅")
        else:
            sys.stdout.write("❌")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    return score

if __name__ == "__main__":
    prompt = "Think out-loud while you answer the question"
    gsm8k_dataset = load_dataset("gsm8k", "main")
    cosine_similarity_score(prompt, gsm8k_dataset["train"])
