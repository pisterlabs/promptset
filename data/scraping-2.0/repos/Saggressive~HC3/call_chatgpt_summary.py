from datasets import load_dataset
import openai
from multiprocessing import Pool
import os
from tqdm import tqdm
import json
openai.api_key="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"

def request_gpt_english_summary(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a helpful assistant that summarizes the article."},
        {"role": "user", "content": f'Help summarize the article.: {text}'},
        ])
    gpt_gen=response["choices"][0]["message"]["content"]
    return gpt_gen

def request_gpt_short_chinese_summary(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "你是概况新闻的得力助手。"},
        {"role": "user", "content": f'在10个字以内概括概况这篇新闻。: {text}'},
        ])
    gpt_gen=response["choices"][0]["message"]["content"]
    return gpt_gen

def request_gpt_title_chinese_summary(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "你是生成文章标题的得力助手。"},
        {"role": "user", "content": f'生成这篇文章的标题。: {text}'},
        ])
    gpt_gen=response["choices"][0]["message"]["content"]
    return gpt_gen

def cnndm_call_gpt(row):
    article = row["article"]
    highlights = row["highlights"]
    try:
        gpt_gen=request_gpt_english_summary(article)
    except:
        try:
            gpt_gen=request_gpt_english_summary(article)
        except:
            gpt_gen="null"
    data = {"article":article,"gpt_gen":gpt_gen,"target":highlights}
    data = json.dumps(data,ensure_ascii=False)
    data = data+"\n"
    return data

def cnndm(mode,sample):
    folder = "cnndm_chatgpt"
    os.makedirs(folder,exist_ok=True)
    dataset = load_dataset("cnn_dailymail",'3.0.0')
    data = dataset[mode]
    data = data.shuffle(seed=42)
    data_selected = []
    for index,i in enumerate(data):
        if index<sample:
            data_selected.append(i)
        else:
            break
    with Pool(processes=128) as p:
        proc_data = p.map(cnndm_call_gpt, tqdm(data_selected))
        file_path = folder + os.sep + f"{mode}.jsonl"
        with open(file_path,"w") as f:
            f.writelines(proc_data)


def xsum_call_gpt(row):
    article = row["document"]
    highlights = row["summary"]
    try:
        gpt_gen=request_gpt_english_summary(article)
    except:
        try:
            
            gpt_gen=request_gpt_english_summary(article)
        except:
            gpt_gen="null"
    data = {"article":article,"gpt_gen":gpt_gen,"target":highlights}
    data = json.dumps(data,ensure_ascii=False)
    data = data+"\n"
    return data

def xsum(mode,sample):
    folder = "xsum_chatgpt"
    os.makedirs(folder,exist_ok=True)
    dataset = load_dataset("xsum")
    data = dataset[mode]
    data = data.shuffle(seed=42)
    data_selected = []
    for index,i in enumerate(data):
        if index<sample:
            data_selected.append(i)
        else:
            break
    with Pool(processes=128) as p:
        proc_data = p.map(xsum_call_gpt, tqdm(data_selected))
        file_path = folder + os.sep + f"{mode}.jsonl"
        with open(file_path,"w") as f:
            f.writelines(proc_data)

def lcsts_call_gpt(row):
    article = row["content"]
    highlights = row["summary"]
    try:
        gpt_gen=request_gpt_short_chinese_summary(article)
    except:
        try:
            
            gpt_gen=request_gpt_short_chinese_summary(article)
        except:
            gpt_gen="null"
    data = {"article":article,"gpt_gen":gpt_gen,"target":highlights}
    data = json.dumps(data,ensure_ascii=False)
    data = data+"\n"
    return data

def lcsts(mode,sample):
    folder = "lcsts_chatgpt"
    os.makedirs(folder,exist_ok=True)
    if mode == "train":
        file_path = "gpt_data/LCSTS_new/train.json"
    elif mode == "validation":
        file_path = "gpt_data/LCSTS_new/dev.json"
    else:
        raise ValueError("mode error")
    data = load_dataset("json",data_files=file_path)["train"]
    data_selected = []
    for index,i in enumerate(data):
        if index<sample:
            data_selected.append(i)
        else:
            break
    with Pool(processes=128) as p:
        data_proc = p.map(lcsts_call_gpt, tqdm(data_selected))
        save_path = folder + os.sep + f"{mode}.jsonl"
        with open(save_path,"w") as f:
            f.writelines(data_proc)

def news2016_call_gpt(row):
    article = row["content"]
    highlights = row["title"]
    try:
        gpt_gen=request_gpt_title_chinese_summary(article)
    except:
        try:
            gpt_gen=request_gpt_title_chinese_summary(article)
        except:
            gpt_gen="null"
    data = {"article":article,"gpt_gen":gpt_gen,"target":highlights}
    data = json.dumps(data,ensure_ascii=False)
    data = data+"\n"
    return data

def news2016(mode,sample):
    folder = "news2016_chatgpt"
    os.makedirs(folder,exist_ok=True)
    if mode == "train":
        file_path = "gpt_data/news2016zh/news2016zh_train.json"
    elif mode == "validation":
        file_path = "gpt_data/news2016zh/news2016zh_valid.json"
    else:
        raise ValueError("mode error")
    data = load_dataset("json",data_files=file_path,split='train', streaming=True)
    data = data.shuffle(seed=42, buffer_size=100)
    data_selected = []
    for index,i in enumerate(data):
        if index<sample:
            data_selected.append(i)
        else:
            break
    with Pool(processes=128) as p:
        proc_data = p.map(news2016_call_gpt, tqdm(data_selected))
        save_path = folder + os.sep + f"{mode}.jsonl"
        with open(save_path,"w") as f:
            f.writelines(proc_data)

if __name__=="__main__":
    mode="train"
    sample=6200
    cnndm(mode,sample)
    xsum(mode,sample)
    lcsts(mode,sample)
    news2016(mode,sample)