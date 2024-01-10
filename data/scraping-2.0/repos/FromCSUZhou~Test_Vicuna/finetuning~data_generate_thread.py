import pandas as pd
import numpy as np
import random
import json
import openai
import threading
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_fixed

openai.api_key = "openai_api_key"
MAX_THREADS = 10

conversation_id = 0  # 对话ID
chunk_size = 500  # 每生成500个对话就写入文件
output_json = []  # 临时存储对话
lock = threading.Lock()


question_json = {
    "question1": "Please put the first question here.",
    "question2": "Please put the second question here.",
    "question3": "Please put the third question here.",
    "question4": "Please put the fourth question here.",
    "question5": "Please put the fifth question here.",
    "question6": "Please put the sixth question here."
}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_gpt_3_5(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=prompt,
        max_tokens=300
    )
    answer = response['choices'][0]['message']['content']
    return answer
    # return json.dumps(["question1", "question2", "question3", "question4", "question5", "question6"])


# 读取origin_question.xlsx中的问题
question_df = pd.read_excel('origin_question.xlsx')
questions = question_df.iloc[:, 0].tolist()

def process_question(question, generated_questions):
    global conversation_id, output_json
    prompt_for_data = "请以足够专业切准确的角度回答这个问题：\n" + generated_questions[question]
    print(prompt_for_data)
    message = [{"role": "user", "content": prompt_for_data}]
    try:
        result = call_gpt_3_5(message)
    except Exception as e:
        return
    print(result)

    with lock:
        # 创建一个新的对话并添加到output_json列表
        conversation = {
            "id": f"identity_{conversation_id}",
            "conversations": [
                {
                    "from": "human",
                    "value": generated_questions[question]
                },
                {
                    "from": "gpt",
                    "value": result
                }
            ]
        }
        output_json.append(conversation)
        conversation_id += 1

        # 如果达到chunk_size，将output_json写入文件并清空列表
        if conversation_id % chunk_size == 0:
            with open(f'output_{conversation_id // chunk_size}.json', 'w', encoding='utf-8') as f:
                json.dump(output_json, f, ensure_ascii=False, indent=4)
            output_json = []


# 使用ThreadPoolExecutor来管理线程
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:

        for i in range(1000):
            selected_questions = random.sample(questions, 3)
            prompt_for_questions = f"Give full play to your imagination and creativity, please turn the following 3 questions into 6 similar questions. " + " ".join(
                selected_questions) + f"Please wrap the 6 generated questions in a json structure like this:" f"{question_json}"
            message = [{"role": "user", "content": prompt_for_questions}]

            try:
                generated_questions_json = call_gpt_3_5(message)
                generated_questions = json.loads(generated_questions_json)

                for question in generated_questions:
                    executor.submit(process_question, question, generated_questions)
            except:
                print("生成的问题不是json格式或API调取出现问题，不进行处理。")

# 如果有剩余对话，将它们写入文件
if output_json:
    with open(f'output_{conversation_id // chunk_size + 1}.json', 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)