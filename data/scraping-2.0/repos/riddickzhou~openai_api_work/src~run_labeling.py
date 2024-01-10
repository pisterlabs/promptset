import os
import json
import openai
import datetime

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = 'gpt-3.5-turbo'
DATA_FILE_PATH = './data/toy_case.json'
OUTPUT_FILE_PATH = './data/'
INSTRUCTION_SYS = '你是一个题库错误梳理专家'
INSTRUCTION_USER = """下面我会输入一些问题以及对应的答案，问题与答案用'-|-'分隔。任务是判断答案是否正确。如果正确返回'YES'，并以专家的口吻作答一遍; 如果不正确，返回'NO'，指出存在的问题并用中文重新生成正确答案。"""
ITER_PER_FILE = 10000
openai.api_key = OPENAI_API_KEY

def get_results(input_messages):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=input_messages,
            temperature=0,
            max_tokens=3500
        )
        return response.choices[0].message['content']
    except Exception as e:  # Catch specific exception
        print(f"Error occurred: {e}")
        return 'Error'


def read_data(file_path, n=-1):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = [json.loads(line.strip()) for line in f.readlines()[:n] or []]
    return [(i['prompt'].strip(), i['response'].strip()) for i in lines]


def build_messages(data):
    return [[
        {"role": "system", "content": INSTRUCTION_SYS},
        {"role": "user", "content": INSTRUCTION_USER},
        {"role": "user", "content": f"{prompt}-|-{response}"}
    ] for prompt, response in data]


def write_results(file, idx, result):
    file.write(f"*** NO.{idx} ***\nGPT_A: {result}\n\n---------------------------------------\n\n")

def main():
    outputs = read_data(DATA_FILE_PATH, n=-1)
    messages = build_messages(outputs)

    f = None
    for idx, message in enumerate(messages):
        if idx % ITER_PER_FILE == 0:
            if f:
                f.close()

            file_number = idx // ITER_PER_FILE
            current_time = datetime.datetime.now()
            timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

            print(f'opening file result_labeling_{file_number}_{timestamp_str}.txt')
            f = open(f'{OUTPUT_FILE_PATH}result_labeling_{file_number}_{timestamp_str}.txt', 'w', encoding='utf-8')

        result = get_results(message)
        print(result)
        write_results(f, idx, result)
    if f:
        f.close()


if __name__ == "__main__":
    main()
