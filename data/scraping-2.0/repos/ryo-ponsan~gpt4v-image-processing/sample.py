import openai
import json
import os
from dotenv import load_dotenv
import tiktoken
import time
import base64
import requests

# .envファイルの内容を読み込む
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 画像をエンコードする関数
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_gpt4v_response_from_encoded_image(base64_image, question, model="gpt-4-vision-preview"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def get_gpt4v_comparison_from_local_images(image_path1, image_path2, question, model="gpt-4-vision-preview"):
    base64_image1 = encode_image(image_path1)
    base64_image2 = encode_image(image_path2)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image1}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image2}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def get_completion_from_messages(messages, model="gpt-3.5-turbo-1106", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

def get_json_from_messages(messages, model="gpt-3.5-turbo-1106", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=model,
        response_format={ "type": "json_object"},
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

def main():
    # Get the current time
    start_time = time.time()
    print("Running...")

    # 入出力ファイル
    input_folder = "./data/"
    output_folder = "./output/"

    # query
    queries_path = "./prompt/queries_sample.json"
    queries = read_json(queries_path)
    query1 = queries['query1']
    query2 = queries['query2']
    system_prompt = queries['system']
    query_json = queries['queryJsonResult']
    # query_json = read_file(queries['queryJsonResultFile'])

    print("=================================1.ローカル画像===============================")
    # ローカル画像に対して
    image_path = input_folder + "grape1.png"

    # Base64文字列の取得
    base64_image = encode_image(image_path)

    # 質問

    response1 = get_gpt4v_response_from_encoded_image(base64_image, query1)

    # 応答を表示
    print(response1)


    # print("=================================2.二枚のローカル画像の比較===============================")

    # image_path2 = input_folder + "grape2.png"
    # image_path3 = input_folder + "grape3.png"
    # response2 = get_gpt4v_comparison_from_local_images(image_path2, image_path3, query2)

    # print(response2)

    print("=================================それぞれの解析結果===============================")
    print("########################### python jsonをリスト形式へ #####################################")

    messages = []
    messages.append({"role": "system", "content": system_prompt})

    print("=================================1のjson結果===============================")
    messages.append({"role": "user", "content": query_json.format(result=response1)})
    response1_json = get_json_from_messages(messages)
    print("message:")
    print(query_json.format(result=response1))
    print("GPT:")
    print(response1_json)


    # print("=================================2のjson結果.2枚の比較===============================")
    # messages2 = []
    # messages2.append({"role": "system", "content": system_prompt})
    # messages2.append({"role": "user", "content": query_json.format(result=response2)})
    # response2_json = get_json_from_messages(messages2)
    # print("message:")
    # print(query_json.format(result=response2))
    # print("GPT:")
    # print(response2_json)

    # Get the current time again
    end_time = time.time()

    # Compute the difference
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()