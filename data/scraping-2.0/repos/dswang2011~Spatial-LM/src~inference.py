
import torch

import os
import argparse
from torch.utils.data import DataLoader

import torch
import pickle
from utils.params import Params
# from torch_geometric.transforms import NormalizeFeatures
import mydataset
from LMs.myinferencer import MyInferencer
import LMs
from utils import util
import openai
import json
import time


# openai.api_key = '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'
openai.api_key = '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'


def parse_args(config_path):
    parser = argparse.ArgumentParser(description='run the model')
    parser.add_argument('--config', dest='config_file', default = config_path)
    return parser.parse_args()


# Function to perform question answering
def question_answer(doc, question):
    prompt = f"Document: {doc} \n Question: based on the given document, {question}? Please only provide the exact answer string (no paraphrasing)." 

    print(prompt)

    try:
        response = openai.Completion.create(
            engine='text-davinci-003', # 'text-davinci-003',	‘gpt-3.5-turbo’	‘gpt-4’
            prompt=prompt,
            max_tokens=50,
            temperature=0,
            n=1,
            stop=None
        )
    except:
        response = openai.Completion.create(
            engine='text-davinci-003', # 'text-davinci-003',	‘gpt-3.5-turbo’	‘gpt-4’
            prompt=prompt,
            max_tokens=50,
            temperature=0,
            n=1,
            stop=None
        )

    # Extract the answer from the API response
    answer = response.choices[0].text.strip()

    return answer


def get_completion(doc, question):
    prompt = f"""
        Based on the given Document, {question} Please only provide the exact answer string (no paraphrasing).

        Use the following format:
        Answer:<answer string>

        Document:
        ```{doc}```
    """
    # print(prompt)

    messages = [{"role": "user", "content":prompt}]

    retry = True
    while retry:
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = messages,
                temperature = 0,
            )
            return response.choices[0].message["content"].strip()
        except:
            print(' sleep and retry')
            retry = True
            time.sleep(10)



if __name__=='__main__':

    # Section 1, parse parameters
    args = parse_args('config/inference.ini') # from config file
    params = Params()   # put to param object
    params.parse_config(args.config_file)
    params.config_file = args.config_file

    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # params.device = torch.device('cpu')
    print('Using device:', params.device)

    # section 2, get the model
    # model = LMs.setup(params).to(params.device)

    #section 3, trainer
    # mytrainer = MyTrainer(params)

    # section 3,data
    # this is usually covered by huggingface models
    # params.output_dir = 'tmp_dir/'
    # for file_path in [
    #     '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_a1_dataset.hf',
    #     # '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_b1_dataset.hf',
    #     # '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_b2_dataset.hf',
    #     # '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_cdip_b3_dataset.hf',
    # ]:
    #     print('-- prepare:', file_path)
    #     params.cdip_path = file_path
    # print('-- load raw:', params.cdip_path)
    mydata = mydataset.setup(params)
    # print('-- finished mapping, now inference:', params.cdip_path)

    # myinferencer = MyInferencer(params)

    # section 6, classifying and decoding labels
    # img_paths,all_preds = myinferencer.inference_for_classification(params, model, mydata)
    # print('finished infering, and prepare to write:',len(img_paths))
    # for img, pred in zip(img_paths,all_preds):
    #     label = model.config.id2label[pred]
    #     util.write_line('class_a.txt', img.strip() + '\t' + str(label))
    # print('--- end of infer for:', file_path)


    # section 7, QA infering and output data
    # myinferencer.inference_for_QA(model,mydata,'docvqa_3.json')

    # section 8, use LLM for inference
    cnt = 0
    res = []

    flag = False
    for inst in mydata.raw_test:
        words = inst['words']
        question = inst['question']
        qID = inst['qID']
        if qID == 64879:
            flag = True
            print('--detected--', qID)
            continue

        if flag == False:
            continue

        answers = inst['answers']

        doc = ' '.join(words)
        # print(doc)
        # print(question)
        # print(qID)

        # answer = question_answer(doc,question)
        answer = get_completion(doc,question)

        # print(question)
        print(answer)

        answer = answer.split("Answer:")[1].strip() 
        # deliver to GPT
        res.append({"questionId":qID, "answer":answer})
        item = str({"questionId":qID, "answer":answer})
        print(item)
        with open('temp.txt','a') as fw:
            fw.write(item + '\n')
        cnt+=1
        if cnt%50==0: 
            print('-sleep-')
            time.sleep(5)

    
    # save it
    res = json.dumps(res)
    with open('gpt_res.json','a') as fw:
        fw.write(str(res))

    
