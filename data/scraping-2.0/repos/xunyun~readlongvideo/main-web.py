# -*- coding: utf-8 -*-

import sys
import os
import openai
import glob
import subprocess
from tqdm import tqdm
import json
import base64

# azure 参数
openai.api_type = "azure"
openai.api_base = "123"
openai.api_version = "2023-03-15-preview"
openai.api_key = '123'


full_prompt_meeting = '''
请一步一步完成下列任务，请只返回直接的结果，不需要附带任何引导语或前缀。：
1、仅处理后面<! !/>内的文字；
2、去掉类似“说话人1 15:29 ”的标记；
3、根据语义的相近程度整理为更合理的段落；
4、去掉冗余的口语表述；
5、使用正式的名称修正提到的机构名；
6、保留尽可能多的原始信息；
7、不要使用项目符号分点论述的风格，不要使用'- '这种分点论述，不要使用'1. 2. 3.'这种论述；
{}
<! 
{}
!/>
'''

#- 请修改每段文字的语病，并保证段落意思不变；
#- 请根据段落主要语境，修改可能是错别字的地方；
#- 使用正式的名称修正提到的机构名；
#- 请修改使每个句子的主语清晰，语法正确；

full_prompt_polish = '''
请根据要求一步一步处理后面<!开始/> <!结束/>内的文字，请只返回直接的结果，不附带任何引导语或前缀。
你处理文字时需要考虑的要求是：
{}
<!开始/>  
{}
<!结束/>
'''

encoded_person_prompt = sys.argv[2]
# person_prompt = base64.b64decode(encoded_person_prompt.encode('utf-8')).decode('utf-8')
new_text = encoded_person_prompt # 这里预留来补充新的修改需求


def openai_result_A(full_prompt):
    openai.api_key = '123'
    response = openai.ChatCompletion.create(
                            engine="gpt-3.5-turbo",
                            prompt=full_prompt,
                            model="gpt-3.5-turbo",
                            messages=[
                                {
                                "role": "user",
                                "content": full_prompt,
                                },
                                    ],
                            temperature=0.38,
                            max_tokens=2336,
                            top_p=0.95,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
    api_json_data = json.dumps(response.to_dict())
    data = json.loads(api_json_data)
    # 从字典中解析出助手的回复
    total_tokens_used = data['usage']['total_tokens']
    assistant_message = data['choices'][0]['message']['content']
    repo = assistant_message
    return repo 

import requests

def openai_result(full_prompt):
    #url = "https://api.openai-proxy.com/v1/chat/completions"
    url = "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": full_prompt
        }
    ],
    "temperature": 0.38,
    "max_tokens": 2336,
    "top_p": 0.95,
    "frequency_penalty": 0,
    "presence_penalty":0,
    })
    headers = {
    'Authorization': '123123',
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    # 从字典中解析出助手的回复
    #total_tokens_used = data['usage']['total_tokens']
    assistant_message = data['choices'][0]['message']['content']
    repo = assistant_message
    return repo 


# 根据完整prmopt获取分类结果
def azure_result(full_prompt):
    response = openai.ChatCompletion.create(
          engine="IFSweb",
            messages = [{"role":"system","content":"你是文案整理专家。"},
                        {"role":"user","content":full_prompt}],
          temperature=0.4,
          max_tokens=3000,
          top_p=0.95,
          frequency_penalty=0,
          presence_penalty=0,
          stop=None)
    
    try:
        content = response.get('choices', [{}])[0].get('message', {}).get('content')
        if not content:
            print(f"Unexpected response structure: {response}")
            return None  # 或者返回其他默认值或错误消息
        repo = content.replace('\n', '').replace(' .', '.').strip()
        return repo
    except Exception as e:
        print(f"Error processing the response: {e}")
        repo = "这段文字处理失败了。"
        return None  # 或者返回其他默认值或错误消息

    # repo = response['choices'][0]['message']['content'].replace('\n', '').replace(' .', '.').strip()
    return repo


def read_text_files(file_path):
    # 使用glob模块找到所有.txt文件
    text_files = glob.glob(os.path.join(file_path, '*.txt'))

    # 创建一个空字典来保存所有文件的内容
    all_texts = {}

    # 对于每个文件
    for text_file in text_files:
        # 打开文件并读取内容
        with open(text_file, 'r', encoding='utf-8') as file:
            row_text = file.read()

            # 将文件名作为键，文件内容作为值，添加到字典中
            all_texts[os.path.basename(text_file)] = row_text

    # 返回包含所有文件内容的字典
    return all_texts


def convert_to_md(input_file, output_file):
    # 命令列表
    command = ["pandoc", input_file, "-o", output_file]
    # 调用 subprocess 的 run 方法来执行命令
    subprocess.run(command)
    return output_file


def convert_to_docx(input_file, output_file):
    # 命令列表
    command = [
        "pandoc",
        input_file,
        "-s",
        "-o",
        output_file,
        "--reference-doc",
        "/download/reference.docx",
    ]
    # 调用 subprocess 的 run 方法来执行命令
    subprocess.run(command)


print ('正在读取文件...')

# 从命令行参数获取文件名
input_file_name = sys.argv[1]
# input_file_name = input("请输入文件名（包括扩展名）：")
base_name = os.path.splitext(input_file_name)[0]  # 获取文件路径和名字，不包括扩展名
output_md_file_name = base_name + ".md"  # 添加新的扩展名

md_file = convert_to_md(input_file_name, output_md_file_name)



print ('正在拆分文件...')
import spliteText  # 在其他脚本中
# spliteText.main(filename)
folder_path, file_count = spliteText.main(output_md_file_name)


print ('正在优化文本...')

# 你可以使用以下代码调用这个函数
file_path =  folder_path 
all_texts_dict = read_text_files(file_path)
folder_name_base = os.path.basename(file_path)  # 输出：TM-20230806100629-AICodesplit


total_files = file_count   # 总文件数，用于计算进度百分比

import threading
import queue

# 创建一个信号量对象，最多允许5个线程同时运行
semaphore = threading.Semaphore(8)

def fetch_data(i, all_texts_dict, folder_name_base, new_text, results_queue):
    with semaphore:
        row_text_iso = all_texts_dict['{}{}.txt'.format(folder_name_base, str(i).zfill(2))]
        formatted_prompt = full_prompt_polish.format(new_text, row_text_iso)
        #text_iso = "测试"
        print ("test{}".format(i))
        #text_iso = openai_result(formatted_prompt)
        text_iso = azure_result(formatted_prompt)
        results_queue.put((i, text_iso))  # 使用索引和结果作为元组放入队列

def process_data_in_threads(all_texts_dict, folder_name_base, new_text, total_files):
    results_queue = queue.Queue()
    threads = []

    for i in range(1, total_files+1):
        t = threading.Thread(target=fetch_data, args=(i, all_texts_dict, folder_name_base, new_text, results_queue))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    # 从队列中收集结果并按索引排序
    results = [results_queue.get() for _ in range(total_files)]
    results.sort(key=lambda x: x[0])  # 根据索引排序

    # 合并结果
    merged_results = "\n\n".join([text_iso for _, text_iso in results])
    
    return merged_results

# 使用新的函数替换原始代码
mid_output_file_name = "{}_mid.txt".format(file_path)
merged_results = process_data_in_threads(all_texts_dict, folder_name_base, new_text, total_files)

with open(mid_output_file_name, "w", encoding='utf-8') as f:
    f.write(merged_results)

# 更新进度
global global_progress
global_progress = 100


print ('正在转换格式...')
# 将处理好的文件转化为docx格式
output_file_name = "{}_final.docx".format(file_path)
convert_to_docx(mid_output_file_name, output_file_name)
print (output_file_name)
print ('已完成！')