from datasets import load_dataset
import openai
from multiprocessing import Pool
import os
from tqdm import tqdm
import json
openai.api_key="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"

def en2zh(row):
    en,zh=row["source"],row["target"]
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "你是个将英文翻译成中文的助手"},
        {"role": "user", "content": f'将下列英文文本翻译成中文。: {en}'},
        ])
        gpt_gen=response["choices"][0]["message"]["content"]
    except:
        gpt_gen="null"

    data = {"source":en,"gpt_gen":gpt_gen,"target":zh}
    data = json.dumps(data,ensure_ascii=False)
    data = data+"\n"
    return data

def request_gpt_english_summary(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a helpful assistant that summarizes the article."},
        {"role": "user", "content": f'Help summarize the article.: {text}'},
        ])
    gpt_gen=response["choices"][0]["message"]["content"]
    return gpt_gen

def delete_null(input_folder,output_folder):
    os.makedirs(output_folder,exist_ok=True)
    file_names = os.listdir(input_folder)
    clean_data = []
    for file_name in file_names:
        path = input_folder + os.sep + file_name
        with open(path) as f:
            data = f.readlines()
            data = [ json.loads(i) for i in data ]
        for i in tqdm(data):
            if i["gpt_gen"]!="null":
                i = json.dumps(i,ensure_ascii=False)
                i = i+"\n"
                clean_data.append(i)
            else:
                # i=en2zh(i)
                # clean_data.append(i)
                article = i["gpt_gen"]
                gpt_gen = request_gpt_english_summary(article)
                i["gpt_gen"] = gpt_gen
                i = json.dumps(i,ensure_ascii=False)
                i = i+"\n"
                clean_data.append(i)
        save_path = output_folder + os.sep + file_name
        with open(save_path,"w") as f:
            f.writelines(clean_data)
    

if __name__=="__main__":
    # delete_null("news2016_chatgpt","news2016_chatgpt_clean")
    # delete_null("cnndm_chatgpt","cnndm_chatgpt_clean")
    # delete_null("xsum_chatgpt","xsum_chatgpt_clean")
    # delete_null("lcsts_chatgpt","lcsts_chatgpt_clean")
    # delete_null("en_de","en_de_clean")
    # delete_null("en_fr","en_fr_clean")
    # delete_null("en_ro","en_ro_clean")
    # delete_null("en_zh","en_zh_clean")
    # delete_null("ours_data/news2016_chatgpt","news2016_chatgpt_clean")
    delete_null("ours_data/xsum_chatgpt","xsum_chatgpt_clean")