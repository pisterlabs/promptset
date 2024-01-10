#logic/index_logic.py
import os
import openai
import time

# OpenAI APIのエンドポイントとAPIキーを設定します
API_KEY = os.environ.get('OPENAI_API_KEY')

def get_ai_response(mes):
    max_retries=3
    retries=0
    res_message="すみません、もう一度お願いします。"
    assumption = "下記は新人からの質問です。優しく丁寧に初学者にもわかりやすく簡潔に説明してください。\n"
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                #model="gpt-4",
                messages=[{"role": "user", "content": assumption + mes}],
                #max_tokens=max_tokens,
                n=1,
                temperature=0.1
            )
            print(assumption + mes)
            print("-----")
            res_message = response.choices[0].message['content'].strip()
            print(res_message)
            return res_message
        except Exception as e:
            retries += 1
            print("Error:", e)
            print("Retrying...")
            time.sleep(3)

    return res_message
