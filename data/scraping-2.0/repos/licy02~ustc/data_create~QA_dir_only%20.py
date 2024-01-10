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

def main(api_key,dir,k):
    openai.api_key = api_key
    filelist=os.listdir(dir)
    p=0
    while p <= len(filelist)-1:
        docu = filelist[p]
        if docu == 'QA':
            p += 1
            continue
        else:
            data_path = dir + "/" + docu
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            lens = []
            if k == 0:
                for j in range(len(data)):
                    t = (int(len(data[j]["content"]) / 50) + 1) * 2
                    if t > 10:
                        t = 10
                    lens.append(t)
                sum_len = list(np.cumsum(lens))
            else:
                for j in range(len(data)):
                    t = k
                    lens.append(t)
                sum_len = list(np.cumsum(lens))
            i = 0
            flag=True
            while flag:
                save_path = dir + "/QA/" + docu
                dirs = dir + "/QA"
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
                key1 = data[i]["name"]
                id1 = data[i]["id"]
                prompt = f"{info}内引用的文本为目标文本，主题为{key1}。你现在是一名中国科学技术大学学生，帮助我对引用的文本提出问题，提出的问题要求满足以下条件：1.提出的问题必须是针对目标文本内容提出的。2.保证提出的问题的答案在内容中能够找到。3.保证提出问题描述清晰，丰富。4.不要生成重复的问题。5.生成的问题要符合人类语言表达习惯。6.只提出针对目标文本的提问，不要生成其他不相关的问题。7.必须生成中文问题，不要生成其他语言的问题，格式为”问“”答“。在你提出问题前，请一定认真阅读上述需要满足的条件列表，这个很重要。现在开始提问吧"
                message = [{"role": "assistant", "content": prompt}]
                completion = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=message,
                    temperature=0.8,
                    max_tokens=800,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=1.0
                )
                response = completion.choices[0].message["content"]
                print(f"response is {response}")
                if "\n\n" or "\n" in response:
                    split_response = re.split("\n\n|\n", response)
                for j in range(int(len(split_response) / 2)):
                    if split_response[2*j][:2] == "问：" and split_response[2*j + 1][:2] == "答：":
                        with open(save_path, "r", encoding="utf-8") as s:
                            size = os.path.getsize(save_path)
                            if size == 0:
                                new_data = []
                            else:
                                new_data = json.load(s)
                        with open(save_path, "w", encoding="utf-8") as save:
                            if len(new_data) < sum_len[i]:
                                new_data.append({"问": split_response[2*j][2:], "答": split_response[2*j + 1][2:], "相关信息": info, "id": id1})
                            print(f"共成功写入{len(new_data)}条")
                            json.dump(new_data, save, indent=4, ensure_ascii=False)
                # 停顿时间
                if i >= len(data)-1 and len(new_data) >= sum_len[i]:
                    flag = False
                    p += 1
                    i = 0
                elif i >= len(data)-1 and len(new_data) < sum_len[i]:
                    i -= 1
                    continue
            continue

if __name__ == "__main__":
    main(api_key=" ",dir=" ",k= )