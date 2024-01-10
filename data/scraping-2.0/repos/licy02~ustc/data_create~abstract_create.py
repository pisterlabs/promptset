import re
import time
import openai
import json
import os
import numpy as np

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)

def get_file_list(dir):
    file=os.listdir(dir)
    filelist=[]
    for item in file:
        filelist.append(item)
    return filelist

def main(api_key,dir,k=0):
    openai.api_key = api_key
    filelist=get_file_list(dir)
    p=0
    while p <= len(filelist)-1:
        docu = filelist[p]
        if docu == 'abstract':
            p += 1
            continue
        else:
            data_path = dir + "/" + docu
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            lens = []


            for j in range(len(data)):
                t = 1
                lens.append(t)
            sum_len = list(np.cumsum(lens))
            i = 0
            flag=True
            while flag:
                save_path = dir + "/abstract/" + docu
                dirs = dir + "/abstract"
                if not os.path.exists(dirs):
                    os.makedirs(dirs)
                if not os.path.exists(save_path):
                    with open(save_path, mode='w', encoding='utf-8') as ff:
                        json.dump([], ff, ensure_ascii=False)
                with open(save_path, mode='r', encoding='utf-8') as s:
                    size = os.path.getsize(save_path)
                    if size == 0:
                        new_data = []
                    else:
                        new_data = json.load(s)
                while len(new_data) >= sum_len[i]:
                    i += 1
                    if i == len(data):
                        break
                info = data[i]["content"]
                id = data[i]["id"]
                t = int(len(data[i]["content"]))
                if t < 50:
                    response = info
                else:
                    prompt = f"依据{info}中的信息，用中文总结主要内容，要求满足以下要求：\n \1.在一句话之内，不需要包含细节，字数在100以内 \n  \2.只需保留事实类的信息，不要保留观点类的信息。保留格式为”“abstract"

                    message = [{"role": "assistant", "content": prompt}]
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0613",
                        messages=message,
                        temperature=0.8,
                        max_tokens=800,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=1.0
                    )
                    response = completion.choices[0].message["content"]
                print(f"response is {response}")
                #if "\n\n" or "\n" in response:
                    #split_response = re.split("\n\n|\n", response)
                #for j in range(int(len(split_response) / 2)):
                    #if split_response[2*j][:2] == "问：" and split_response[2*j + 1][:2] == "答：":
                with open(save_path, "r", encoding="utf-8") as s:
                    size = os.path.getsize(save_path)
                    if size == 0:
                        new_data = []
                    else:
                        new_data = json.load(s)
                with open(save_path, "w", encoding="utf-8") as save:
                    if len(new_data) < sum_len[i]:
                        new_data.append({"abstract": response, "content": info, "id": id})
                    print(f"共成功写入{len(new_data)}条")
                    json.dump(new_data, save, indent=4, ensure_ascii=False)
                # 停顿时间

                if i >= len(data)-1 and len(new_data)>=sum_len[i]:
                    flag = False
                    p += 1
                    i = 0
                elif i >= len(data)-1 and len(new_data)<sum_len[i]:
                    i = 0
            continue

if __name__ == "__main__":
    main(api_key=" ",dir="  ")