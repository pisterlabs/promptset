from audioop import add
import os
from turtle import done
import openai
import requests
import time
import difflib
import nltk
import json
from io import StringIO

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Engine.list()

path = "data/rep/no_gap_context"
done_f_path = "done_pre_suff.txt"

def query_openai(prefix, suffix):
    response = openai.Completion.create(
        engine="code-davinci-002", 
        temperature=0.8,
        prompt=prefix,
        suffix=suffix,
        max_tokens=100,
        best_of=21,
        n=20,
        logprobs=1,
    )
    return response

def get_candidates(response):
    count = 0
    candidates = []
    entropies = []
    for result in response["choices"]:
        count += 1
        candidate = result["text"]
        candidates.append(result["text"])
        token_logprobs = result.logprobs.token_logprobs
        entropy = sum(token_logprobs)*(-1)
        entropies.append(entropy) 

    return candidates, entropies

def save_data(candidates, entropies, file):
    base_path = "patch-data/prefix-suffix/rep/no_gap"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    full_path = base_path + "/" + file
    f = open(full_path, "w+")
    for cand,ent in zip(candidates, entropies):
        f.write(json.dumps({"candidate": cand, "entropy": ent})+"\n")
    f.close()

def get_files(path):
    files = []
    for file in os.listdir(path):
        files.append(file)
    return files

def read_context(path, file):
    f = open(os.path.join(path, file))
    data = json.load(f)
    prefix = data["prefix"]
    suffix = data["suffix"]
    return prefix, suffix 

def complete_files(done_f_path):
    f = open(done_f_path, "r")
    done_files = []
    for line in f:
        done_files.append(line.strip())
    return done_files


count = 1
files = get_files(path)
done_files = complete_files(done_f_path)
for file in files:
    if file in done_files:
        print(file, "in done files")
        continue
    print("Running for:", file)
    if count % 5 == 0:
        print("Sleeping on file {}".format(file))
        time.sleep(61)
    prefix, suffix = read_context(path, file)
    try:
        response = query_openai(prefix, suffix)
        candidates, entropies = get_candidates(response)
        save_data(candidates, entropies, file) 
        done_f = open(done_f_path, "a+")
        done_f.write(file+"\n")
        done_f.close()
        count += 1
    except:
        print("Timed out or error on file {}".format(file))


