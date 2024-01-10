# -*- coding: utf-8 -*-

import os
import openai
import random
import time
import codecs
import sys
import re
import json
import regex

import chardet

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
import io
from contextlib import contextmanager

@contextmanager
def utf8_stdout():
    original_stdout = sys.stdout
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')
    try:
        yield
    finally:
        sys.stdout = original_stdout



def read_file_with_detected_encoding(file_path):
    # 使用chardet检测文件编码
    def detect_encoding(file_path):
        with open(file_path, 'rb') as file:
            return chardet.detect(file.read())['encoding']

    # 获取检测到的编码
    detected_encoding = detect_encoding(file_path)

    # 尝试使用检测到的编码读取文件
    try:
        with open(file_path, 'r', encoding=detected_encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        pass  # 如果发生解码错误，尝试使用其他编码

    # 尝试使用其他编码
    for encoding in ['utf-8', 'iso-8859-1', 'gb18030']:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            pass

    # 所有尝试均失败时抛出异常
    raise UnicodeDecodeError(f"无法使用任何已知编码解码文件: {file_path}")





def extract_complete_json(response_text):
    # 使用正则表达式模式匹配嵌套的JSON结构，使用`regex`模块
    json_pattern = r'(\{(?:[^{}]|(?1))*\})'
    matches = regex.findall(json_pattern, response_text)
    if matches:
        try:
            # 尝试解析每个匹配项以找到第一个有效的JSON
            for match in matches:
                json_data = json.loads(match)
                # 返回第一个有效的JSON数据
                return json_data
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
    return None

def gpt_analyze_resume(resume_txt_path, prompt_filename):
    # 记录分析开始时间
    start_time = time.time()
    print("现在时间是:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # OpenAI的API密钥列表，我们的代码会从以下的keys中随机选择一个key
    api_keys = [
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
    ]

    # 假设resume_txt_path位于'Text_Conversions'目录中
    # 我们需要向上跳两级以获取基本目录
    base_dir = os.path.dirname(os.path.dirname(resume_txt_path))
    
    # 构建提示文本的路径
    # 它位于'Analysis_Results/Prompt_Texts'中
    prompt_txt_path = os.path.join(base_dir, "Analysis_Results", "Prompt_Texts", f'{prompt_filename}.txt')


    # 根据提示文本确定JSON的输出目录
    if 'Talent_Portrait' in prompt_filename:
        output_json_dir = "GPT_Talent_Portraits"
        default_json_file = "talent_profile.json"

    elif 'Job_Matching' in prompt_filename:
        output_json_dir = "GPT_Job_Matching"
        default_json_file = "job_match.json"

    elif 'Basic_Infos' in prompt_filename:
        output_json_dir = "Basic_Infos"
        default_json_file = "basic_info.json" 

    else:
        raise ValueError("未知的提示类型。")

    
    # 如果目录不存在，则创建输出目录
    json_output_path = os.path.join(base_dir, "Analysis_Results", output_json_dir)
    os.makedirs(json_output_path, exist_ok=True)

    # 读取简历文本
    # 读取简历文本，忽略编码错误
    with open(resume_txt_path, 'r', encoding='utf-8') as file:
        resume_text = file.read()

    
    # resume_text = read_file_with_detected_encoding(resume_txt_path)

    # 读取提示文本
    with open(prompt_txt_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    # prompt = read_file_with_detected_encoding(prompt_txt_path)

    # 准备用于OpenAI API的消息
    message = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"我这里有一份简历，我需要获取其中的一些信息。简历如下：{resume_text}"}
    ]

    # 设置API端点
    os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1/chat/completions"

    # 随机选择一个API密钥
    openai.api_key = random.choice(api_keys)
    print(f"【{prompt_filename}】api_key:", openai.api_key)

    # 发送OpenAI API请求的函数
    def openai_request(message):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=message,
                temperature=0.01,
                max_tokens=3000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"发生错误: {e}")
            return None

    # 发送请求并获取答案
    answer = openai_request(message)

    # 提取响应中的JSON
    extracted_json = extract_complete_json(answer)
    
    # 如果成功提取JSON，则将其保存到文件中
    if extracted_json is not None:
        json_file_name = os.path.splitext(os.path.basename(resume_txt_path))[0] + ".json"
        json_file_path = os.path.join(json_output_path, json_file_name)
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(extracted_json, json_file, ensure_ascii=False, indent=2)
        print(f"【{prompt_filename}】JSON输出已保存到 {json_file_path}")
    else:
        print(f"【{prompt_filename}】未找到JSON响应，将使用默认JSON文件。")

         # 获取当前脚本所在的目录
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 构建默认JSON文件的路径
        default_json_path = os.path.join(script_dir, "default_json", default_json_file)
        
        # 读取默认JSON文件
        with open(default_json_path, 'r', encoding='utf-8') as file:
            default_json_data = json.load(file)

        # 保存默认JSON到分析结果目录
        json_file_name = os.path.splitext(os.path.basename(resume_txt_path))[0] + ".json"
        json_file_path = os.path.join(json_output_path, json_file_name)
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(default_json_data, json_file, ensure_ascii=False, indent=2)
        print(f"【{prompt_filename}】默认JSON输出已保存到 {json_file_path}")
    


    # 在需要更改编码的地方使用with语句
    with utf8_stdout():
        # 这里的输出将使用UTF-8编码
    # 输出答案
        sys.stdout.write(f"【{prompt_filename}】" + "\n" + answer + "\n")
        sys.stdout.flush()


        print("现在时间是:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # 打印分析所花费的时间
        print(f"分析耗时 {time.time() - start_time} 秒")
        sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("命令行传入错误")
        sys.exit(1)

    resume_txt_path = sys.argv[1]
    prompt_filename = sys.argv[2]
    gpt_analyze_resume(resume_txt_path, prompt_filename)