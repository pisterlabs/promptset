import json
import os
import re
import time_unit
import warnings
from random import sample
import random
import openai
import pandas as pd
from overrides import overrides
from tqdm import tqdm, trange
import fire

warnings.filterwarnings("ignore")


def fields_to_instance(fields, train=True):

    src = fields['src'].to_string(index=False)
    context_type = fields['context_type'].to_string(index=False)
    c_l = fields['left_context'].to_string(index=False)
    c_r = fields['right_context'].to_string(index=False)
    typed_seq = fields['typed_seq'].to_string(index=False)

    if train:
        target = fields['target'].to_string(index=False)
    else:
        target = ''
    return src, context_type, c_l+' [MASK] '+c_r, typed_seq, target



def inference(
    api_key,
    prompt_sample_path,
    inference_data_path,
    output_name,
    random_seed=4,
    debug = True, 
    language = 'zh-en',
    shot = 1,
    dataset_size=None
):

    language_type = {'de-en':0, 'en-de':1, 'zh-en':2, 'en-zh':3}

    openai.api_key = api_key
    random.seed = random_seed


    train_data = pd.read_json(prompt_sample_path,lines=True)
    test_data = pd.read_json(inference_data_path)
    test_data['generation'] = ''

    if debug:
        test_data = test_data.sample(5)
    
    if dataset_size != None:
        test_data = test_data.sample(dataset_size, replace=True)
    
    for id in trange(len(test_data)):
        prompt = ''

        # sample some data out
        examplers = train_data.sample(shot)
        prompt_template = ''

        if language_type[language]==0:
            prompt_template = "Translate this German to English by replacing [MASK] with only one word, one punctuation or nothing.\nSource: {}\nTarget: {}\nAnswer prefix: {}.\n"
        elif language_type[language]==1:
            prompt_template = "Translate this English to German by replacing [MASK] with only one word, one punctuation or nothing.\nSource: {}\nTarget: {}\nAnswer prefix: {}.\n"
        elif language_type[language]==2:
            prompt_template = "Translate this Chinese to English by replacing [MASK] with only one word, one punctuation or nothing.\nSource: {}\nTarget: {}\nAnswer prefix: {}.\n"
        elif language_type[language]==3:
            prompt_template = "Translate this English to Chinese by replacing [MASK] with only one word, one punctuation or nothing.\nSource: {}\nTarget: {}\nAnswer prefix: {}.\n"
        else:
            raise NotImplementedError
        

        for i in range(shot):

            exampler = examplers[i:i+1]

            ref, context_type, context, typed_seq, target = fields_to_instance(exampler)
            prompt += (prompt_template+'Answer: {}\n').format(ref, context, typed_seq, target)
        fields = test_data[id:id+1]
        ref, context_type, context, typed_seq, target = fields_to_instance(fields,train=False)
        prompt += prompt_template.format(ref, context, typed_seq)

        if debug:
            print(prompt)
            '''
            if len(prompt.split(" "))>max_token:
                print(max_token)
                max_token = len(prompt.split(" "))
            pass
            '''
        
        while True:
            try:
                gen = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60,
                    temperature=0.7
                )
            except:
                time_unit.sleep(10)
                continue
            break
        if debug:
            print('Generation',gen)
        time_unit.sleep(1)
        test_data.iloc[id, -1] = gen['choices'][0]['message']['content'].strip()
    # test_data.to_json('/data/hulab/ywu676/wlac-23/gpt/'+output_name, orient='records', indent=4, ensure_ascii=False)
    output_name = output_name.format(language, shot)
    test_data.to_json(output_name, indent=4, orient='records', force_ascii=False)
        


if __name__ == '__main__':
    fire.Fire(inference)