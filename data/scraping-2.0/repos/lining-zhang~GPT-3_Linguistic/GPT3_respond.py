'''
Usage:
    $ python GPT3_respond.py "tense"
'''
import sys
import time
import pandas as pd
import openai
from prompts import Tense_prompt, Subj_num_prompt, Obj_num_prompt, Tense_prompt_general, Subj_prompt_general, Obj_prompt_general

openai.api_key = "your_api_key" # get OpenAI API key

prompt_dict =  {'tense': Tense_prompt,
                'tense_prompt': Tense_prompt_general,
                'subj': Subj_num_prompt,
                'subj_prompt': Subj_prompt_general,
                'obj': Obj_num_prompt,
                'obj_prompt': Obj_prompt_general}

def load_data(path):
    results = []
    with open(path) as f:
        for line in f:
            label, sentence = line.split("\t")[0], line.split("\t")[1]
            results.append([sentence, label])
    return results

def write_csv_file(temp, prompt_type, sentence_list, label_list, response_list):
    df = pd.DataFrame({"sentence": sentence_list,
                       "label": label_list,
                       "GPT3_response": response_list})
    df.to_csv('result/'+ temp + prompt_type + "_result.csv", index=False)

def main(temp, prompt_type, prompt, path):
    sentence_label = load_data(path)
    sentence_list = []
    label_list = []
    response_list = []

    if prompt_type in ['tense', 'tense_prompt']:
        for i, s_l in enumerate(sentence_label):
            if (i + 1) % 100 == 0:
                print(f"Getting GPT-3 response for record {(i + 1)}...") 
            sentence = s_l[0]
            label = s_l[1]
            sentence_list.append(sentence)
            label_list.append(label)

            response = openai.Completion.create(engine="text-davinci-002",
                                                # prompt=Tense_prompt_general(sentence),
                                                prompt=prompt(sentence),
                                                temperature=float(temp),
                                                max_tokens=50)
            response = response["choices"][0]["text"]
            response_list.append(response)
            time.sleep(1)

    if prompt_type in ['subj', 'subj_prompt']:
        for i, s_l in enumerate(sentence_label):
            if (i + 1) % 100 == 0:
                print(f"Getting GPT-3 response for record {(i + 1)}...") 
            sentence = s_l[0]
            label = s_l[1]
            sentence_list.append(sentence)
            label_list.append(label)

            response = openai.Completion.create(engine="text-davinci-002",
                                                # prompt=Subj_prompt_general(sentence),
                                                prompt=prompt(sentence),
                                                temperature=float(temp),
                                                max_tokens=50)
            response = response["choices"][0]["text"]
            response_list.append(response)
            time.sleep(1)

    if prompt_type in ['obj', 'obj_prompt']:
        for i, s_l in enumerate(sentence_label):
            if (i + 1) % 100 == 0:
                print(f"Getting GPT-3 response for record {(i + 1)}...") 
            sentence = s_l[0]
            label = s_l[1]
            sentence_list.append(sentence)
            label_list.append(label)

            response = openai.Completion.create(engine="text-davinci-002",
                                                # prompt=Obj_prompt_general(sentence),
                                                prompt=prompt(sentence),
                                                temperature=float(temp),
                                                max_tokens=50)
            response = response["choices"][0]["text"]
            response_list.append(response)
            time.sleep(1)

    print("Writing results to csv file...")
    write_csv_file(temp, prompt_type, sentence_list, label_list, response_list)

if __name__ == '__main__':
    path_dict = {
    "tense": "data/tense_data.txt",
    "tense_prompt": "data/tense_data.txt",
    "subj": "data/subj_num_data.txt",
    "subj_prompt": "data/subj_num_data.txt",
    "obj": "data/obj_num_data.txt",
    "obj_prompt": "data/obj_num_data.txt"
    } # specify data path

    temp = sys.argv[1]
    prompt_type = sys.argv[2]

    path = path_dict[prompt_type]
    prompt = prompt_dict[prompt_type]

    main(temp, prompt_type, prompt, path)

