import os
import requests
import json
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
import openai

'''
爬取 planfit 资源，将健身关键字翻译为中文并存储到json文件中。

作者：kevintao
日期：2023-07-13
更新：2023-07-14
'''

CUSTOM_DICTIONARY_FILE = 'custom_dictionary_openai.json'  # 存储健身关键字的文件
TRANSLATED_ITEMS_FILE = 'translated_succeed_items.txt'
FAILED_ITEMS_FILE = 'translated_failed_items.txt'

# Load keys from config.json
with open('config.json', 'r') as f:
    config = json.load(f)
    openai.api_key = config['open_ai_api_key']
    openai.api_base = config['open_ai_api_base']

def scrape_planfit():
    base_url = 'https://guide.planfit.ai'
    start_url = base_url

    response = requests.get(start_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    parts = soup.find_all('p', class_='Desktop_partName__mZoSZ')
    total_num_part = len(soup.find_all('p', class_='Desktop_partName__mZoSZ'))
    total_num_item = len(soup.find_all('a', class_='Desktop_trainingModelItemNameText__87kRl'))

    print(f"Total number of Part: {total_num_part}")
    print(f"Total number of Item: {total_num_item}")

    print("开始爬取...")

    dictionary = {}  # 创建一个字典来保存所有的 part_name 和 item_name

    global_item_index = 0  
    for part_index, part in enumerate(parts, start=1):
        print(f"开始爬取 part {part}...")

        part_name = part.text.strip()
        dictionary[part_name] = ""  # 添加 part_name 到字典

        parent_element = part.parent

        trainingModelItemNameTexts = parent_element.find_all('a', class_='Desktop_trainingModelItemNameText__87kRl')

        for item_index, trainingModelItemNameText in enumerate(trainingModelItemNameTexts, start=1):

            target_href = trainingModelItemNameText['href']
            item_name = trainingModelItemNameText.text.strip()

            dictionary[item_name] = ""  # 添加 item_name 到字典

            global_item_index += 1  

            print(f"Scraping {global_item_index}/{total_num_item}... {part_name}/{item_name}...")

    print("爬取结束.")
    
    # 在爬虫结束后，将 dictionary 保存到一个 JSON 文件中
    with open(CUSTOM_DICTIONARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
    print(f"成功保存健身关键字到 {CUSTOM_DICTIONARY_FILE} 文件中.")

def translate_dictionary():
    # 打开 JSON 文件并读取 dictionary
    with open(CUSTOM_DICTIONARY_FILE, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)

    # 创建或打开一个文件来记录已经翻译的关键字
    if os.path.exists(TRANSLATED_ITEMS_FILE):
        with open(TRANSLATED_ITEMS_FILE, 'r') as f:
            translated_items = f.read().splitlines()
    else:
        translated_items = []

    keys = list(dictionary.keys())  # 获取所有的关键字
    pbar = tqdm(total=len(keys), ncols=70)  # 创建一个进度条

    # 对 dictionary 中的每个键进行翻译
    for key in keys:
        if key in translated_items:  # 如果关键字已经被翻译过，就跳过
            pbar.update(1)  # 更新进度条
            continue

        for retry in range(3):  # 重试3次
            try:
                print(f"开始翻译 {key} ...")

                # 使用 OpenAI 的 API 进行翻译
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "请你担任翻译专家，我将提供给你「健身」领域关于「动作」和「身体部位」的词汇，请你将其翻译为中文，只提供英文对应的中文，不要拼音和任何解释。",
                        },
                        {
                            "role": "user",
                            "content": key,
                        },
                    ]
                )
                
                # 提取模型生成的消息，即译文
                translation = response['choices'][0]['message']['content']
                
                dictionary[key] = translation
                print(f"成功翻译 {key} 为 {translation}")
                time.sleep(1)  # 每次请求后暂停1秒，避免频繁请求

                # 将翻译后的 dictionary 保存回 JSON 文件
                with open(CUSTOM_DICTIONARY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(dictionary, f, ensure_ascii=False, indent=4)

                # 将已经翻译的关键字写入文件
                with open(TRANSLATED_ITEMS_FILE, 'a') as f:
                    f.write(key + '\n')

                break  # 如果成功，就跳出重试循环
            except Exception as e:
                print(f"翻译 {key} 失败：{e}")
                if retry < 2:  # 如果还没有重试3次，就继续重试
                    print("重试...")
                    time.sleep(5)  # 暂停5秒，避免频繁请求
                else:  # 如果已经重试3次，就放弃
                    print("放弃。")
                    # 将失败的关键字写入文件
                    with open(FAILED_ITEMS_FILE, 'a') as f:
                        f.write(key + '\n')
                    break

        pbar.update(1)  # 更新进度条

    pbar.close()  # 关闭进度条
    print(f"成功将健身关键字翻译为中文并保存到 {CUSTOM_DICTIONARY_FILE} 文件中。")


if __name__ == '__main__':
    # scrape_planfit()
    translate_dictionary()
