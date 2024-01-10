#!/usr/bin/env python

import openai
from dotenv import load_dotenv
import os
import json
import logging
import time

def formatt_responses(responses):
    a = []
    return a

def main(prompt_file_name):
    # ---設定ファイルの初期化---
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    engine_3 = os.getenv('ENGINE_GPT_3')
    engine_4 = os.getenv('ENGINE_GPT_4')

    with open(prompt_file_name, 'r') as file:
        prompt = file.read()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    # ---設定ファイルの初期化---
    
    # APIリクエスト
    try:
        logging.info('Starting API request...')
        start_time = time.time()
        responses = []
        prompts = [prompt[i:i+4096] for i in range(0, len(prompt), 4096)]  # promptを4096文字ごとに分割
        for prompt in prompts:
            response = openai.ChatCompletion.create(
                model=engine_3,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Please summarize the following text: {prompt}"}
                ]
            )
            responses.append(response)
        end_time = time.time()
        logging.info('Finished API request in %.2f seconds.', end_time - start_time)
    except Exception as e:
        logging.error('Error occurred during API request: %s', e)
        raise

    # APIレスポンスを整形
    formatted_responses = [json.dumps(response['choices'][0]['message'], indent=4, ensure_ascii=False) for response in responses]

    # APIレスポンスの実行結果の出力
    with open('result.json', 'w') as json_file:
        json.dump([response['choices'][0]['message'] for response in responses], json_file, indent=4, ensure_ascii=False)
    with open('result.txt', 'w') as txt_file:
        txt_file.write("\n".join([response['choices'][0]['message']['content'] for response in responses]))

    return formatted_responses  # レスポンスのリストを返す
