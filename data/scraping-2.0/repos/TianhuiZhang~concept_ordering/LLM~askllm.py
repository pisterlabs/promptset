import json
import openai
import time
from tqdm import tqdm
from aligner import Aligner
from collections import Counter

openai.api_key = ""
aligner = Aligner()

def gpt35_reply(concept_set, n, delay=5):
    prompt = f"Generate a sentence containing all the concepts in the concept set: {concept_set}" 
    chat_record = [{"role": "user", "content": prompt}]
    #gpt3.5-turbo
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=chat_record, max_tokens=32, n=n, temperature=0.3)
    sentences = [c.message["content"] for c in completion.choices]
    return sentences


def gpt3_zeroshot_reply(concept_set, n=1, delay=2):
    prompt = f"Generate a sentence with these concepts: {concept_set}."
    res = openai.Completion.create(model="curie",prompt=prompt,max_tokens=32,temperature=1)
    
    sentences = [res['choices'][0]["text"].strip("\n").split("\n")[0].split(". ")[0]]
    print(sentences)
    #time.sleep(delay)
    return sentences

def gpt3_reply(concept_set, n=1, delay=2):
    # curie:ft-personal-2023-05-05-16-01-25 finetune curie original ordering
    # babbage:ft-personal-2023-05-07-00-39-51 finetune babbage original ordering
    # babbage:ft-personal-2023-05-07-01-01-08 finetune babbage example ordering
    # curie:ft-personal-2023-05-07-01-04-33 finetune curie example ordering
    prompt = concept_set +" ->"
    res = openai.Completion.create(model="curie:ft-personal-2023-05-07-01-04-33",prompt=prompt,max_tokens=32,temperature=0)
    
    sentences = [res['choices'][i]["text"].strip().split("\n")[0] for i in range(n)]
    print(sentences)
    #time.sleep(delay)
    return sentences


    

def process_concept_set(concept_set, n):
    #sentences = gpt35_reply("{"+concept_set+"}", n)
    sentences = gpt3_zeroshot_reply("{"+concept_set+"}", n)
    #sentences = gpt3_reply(concept_set)
    candidate = {"src": concept_set, "sentences": [], "orders": []}
    for sent in sentences:
        plan, _, _, _ = aligner.align(concept_set.split(), sent, multi=False, distance=1)
        if len(plan) == len(concept_set):
            candidate["sentences"].append(sent)
            candidate["orders"].append(plan)
            
    with open("temp.jsonl", "a+") as f:
        f.write(json.dumps(candidate) + '\n')
    return candidate


def generate_sentences(src_file, n):
    concept_sets = []
    with open(src_file, 'r', encoding='utf-8') as f:
        for line in f:
            concept_set = line.strip()
            if concept_set not in concept_sets:
                concept_sets.append(concept_set)
    candidates = [process_concept_set(concept_set, n) for concept_set in tqdm(concept_sets)]
    return candidates



def generate_json(split):
    source_path = "../commongen/dataset/commongen." + split + ".src_alpha.txt"
    target_path = "../commongen/dataset/commongen." + split + ".tgt.txt"
    out_path = split + "_original.jsonl"

    with open(source_path) as source, open(target_path) as target, open(out_path, "w") as output:
        source_lines = source.readlines()
        target_lines = target.readlines()

        assert len(source_lines) == len(target_lines)

        for i in range(len(source_lines)):
            source_line = source_lines[i].strip()
            target_line = target_lines[i].strip()

            out = {"prompt": "Generate a sentence containing all the concepts in the concept set: {"+source_line+"} ", "completion": target_line}
            json.dump(out, output)

            if i != len(source_lines) - 1:
                output.write("\n")


import jsonlines

def get_sentences_orders(filename, src, n):
    sentences = []
    orders = []

    with jsonlines.open(filename) as reader:
        for obj in reader:
            if obj['src'] == src:
                sentences.extend(obj['sentences'][:n])
                orders.extend(obj['orders'][:n])
                if len(sentences) >= n:
                    break

    return sentences, orders


def extract_predictions(src_file, predict_file,res_sentences,res_orders):
    sentences = []
    orders = []
    previous_set = None
    count =0
    with open(src_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            concepts = line.strip()
            if concepts != previous_set:
                if previous_set is not None:
                    n_sent,n_order = get_sentences_orders(predict_file, previous_set, 1)
                    sentences.extend(n_sent*count)
                    orders.extend(n_order*count)
                    
                count = 0
                previous_set = concepts
            
            count +=1
        if previous_set is not None:
            n_sent,n_order = get_sentences_orders(predict_file, previous_set, 1)
            sentences.extend(n_sent*count)
            orders.extend(n_order*count)
            
    print(len(sentences),len(orders))
            
    with open(res_sentences, 'w', encoding='utf-8') as file1, open(res_orders, 'w',encoding='utf-8') as file2:
        for item1, item2 in zip(sentences, orders):
            addressed_sentence = str(item1).split("\n")[0]
            file1.write(addressed_sentence + '\n')
            file2.write(str(item2) + '\n')



if __name__ == '__main__':    
    #Upload file
    #res = openai.File.create(file=open("train_original_prepared.jsonl", "rb"),purpose='fine-tune')
    #Create fine-tune model
    res = openai.FineTune.create(training_file="file-N63zsDwSBU4LSxvoWiZ1Ub3v",model="babbage")
    #check the status
    #res = openai.FineTune.retrieve(id ="ft-DLzLpXiSVgJOyGhanf1LJkMs")
    #Predict single
    #res = openai.Completion.create(model="babbage:ft-personal-2023-05-07-00-39-51",prompt="stand field look ->",max_tokens=32,temperature=0.2)
    
    print(res)
    
    
    #Generate jsonl files from model
    """folder = "gpt3curie_zeroshot"
    src_file =  "../commongen/dataset/commongen.test.src_alpha.txt"
    order_file = "../commongen/dataset/commongen.test.src_sentence.txt"
    temp_file = f"../testset/experiment3/{folder}/res.jsonl"
    res_sentences = f"../testset/experiment3/{folder}/sentences.txt"
    res_orders = f"../testset/experiment3/{folder}/orders.txt"
    candidates = generate_sentences(order_file,1)
    with open(temp_file,'w', encoding='utf-8') as f:
        for i in candidates:
            f.write(json.dumps(i) + '\n')
    #Extract result files from jsonl
    extract_predictions(order_file,temp_file, res_sentences,res_orders)"""