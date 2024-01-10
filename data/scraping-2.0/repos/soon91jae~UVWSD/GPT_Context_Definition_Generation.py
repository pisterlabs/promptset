import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import openai
import time
from nltk.corpus import wordnet as wn
import nltk
from tqdm import tqdm

openai.api_key = "" # Add your API Key here

@retry(wait=wait_random_exponential(min=30, max=180), stop=stop_after_attempt(3))
def get_target_word_pos(text, word):
    word_index = text.index(word)


    prev_text = text[:word_index]; next_text = text[word_index+len(word):]
    prev_tokens = nltk.tokenize.word_tokenize(prev_text); next_tokens = nltk.tokenize.word_tokenize(next_text)
    tokens = prev_tokens + [word] + next_tokens

    token_index = len(prev_tokens)
    
    tag = nltk.pos_tag(tokens)
    nltk_pos = tag[token_index][1]
    
    if nltk_pos.startswith('NN'): pos = 'n';
    elif nltk_pos.startswith('V'): pos = 'v';
    elif nltk_pos.startswith('JJ'): pos = 'adj';
    elif nltk_pos.startswith('RB'): pos = 'adv';
    else: pos = 'n'
    return tag[token_index], pos

sense_definitions = []

def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def data_loader(data_file_path, gold_file_path = None):
    
    text_data = {}
    
    fin_data = open(data_file_path)
    for data_index, line in enumerate(fin_data):
        line = line.strip()
        if not line: continue
        
        cols = line.split('\t')
        target_word = cols[0]; context = cols[1]
        candidates = cols[2:]
        
        sense_definitions = []
        target_senses = wn.synsets(target_word)
        for synset in target_senses:
            if synset.pos() == 'n':
                sense_definition = synset.definition().split(';')[0]
                sense_definitions.append(sense_definition)
            
        text_data[data_index] = {'target_word': target_word,
                                 'sense_definitions': sense_definitions,
                                 'context': context,
                                 'candidates': candidates}
    fin_data.close()
    
    
    if gold_file_path:
        fin_gold = open(gold_file_path)
        for gold_index, line in enumerate(fin_gold):
            line = line.strip()
            if not line.strip(): continue
            
            gold = line.strip()
            text_data[gold_index]['gold'] = gold
            
    return text_data

wait_time = 0.1
batch_size = 20

data_file_path = "datapath/train/train_v1/train.data.v1.txt"
gold_file_path = "datapath/train/train_v1/train.gold.v1.txt"


text_data = data_loader(data_file_path)
total_len = len(text_data)
print(total_len)
start_pos = 0
i = 0

output_filename = "GPT_Context_Definitions.txt"

if os.path.isfile(output_filename):
    file = open(output_filename, 'r')
    Lines = file.readlines()
    
    contextList = []
    for l in Lines:
        try:
            l = l.strip("\n")
            if (len(l) > 0) and (len(l.split("\t")) == 3):
                contextWord = l.split("\t")[1]
                contextList.append(contextWord)
        except Exception as e:
            print(e)
            print(l)
            print(l.split("\t"))
            print("prev l", l)
        prev_l = l
else:
    contextList = []
    
print(len(contextList))



for idx, data in tqdm(text_data.items()):
    if data['context'] in contextList:
        contextList.remove(data['context'])
    else:
        _, pos = get_target_word_pos(data['context'], data['target_word'])
        #print(data['target_word'], data['context'], pos)
        prompt = "Define \"{}\" in {}. {} ({}):".format(data['target_word'], data['context'], data['target_word'], pos)
        #prompt = "{} ({}):".format(data['target_word'], pos)
        #print(prompt)
        try:
            response = completions_with_backoff(model="text-davinci-002", prompt=prompt, temperature=1.0, max_tokens=64)
            # response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.7, max_tokens=64)
            definition = response["choices"][0]['text']
            strt_idx = definition.find("\n\n")
            definition = definition[strt_idx+2:]
            format_def = definition.replace("\n", " ")
            # print("{}\t{}\t{}\n".format(data['target_word'], data['context'], format_def))
            with open(output_filename, "a") as out_file:
                out_file.write("{}\t{}\t{}\n".format(data['target_word'], data['context'], format_def))
        except Exception as e:
            print(openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.7, max_tokens=64))
            raise("Exception", e)

        #print("Sleeping for {} secs".format(wait_time))
        time.sleep(wait_time)

        i += 1
        # break

# response = openai.Completion.create(model="text-davinci-003", prompt="Define pollosimo gallinacean", temperature=0.7, max_tokens=128)
# print(response)
print("Done!")
