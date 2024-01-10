import openai
import json


def test_gpt_4():
    import os
    from openai import OpenAI
    client = OpenAI()
    OpenAI.api_key = os.getenv('OPENAI_API_KEY')
    input_issue_name = "week password rule"
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': f'多數情況下,\"{input_issue_name}\"的 CVSSv3 分數最可能是多少? 只回答我數字就好'}
        ],
        temperature=0
    )

    print(completion.choices[0].message.content)


def read_ask_gpt3(input_string):
    # 設定 API 金鑰
    openai.api_key = "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"

    # 定義問題
    question = input_string

    # 設定 GPT-3 模型的引擎 ID
    model_engine = "gpt-3.5-turbo-instruct"

    # 呼叫 OpenAI API 並取得答案
    response = openai.Completion.create(
        engine=model_engine,
        prompt=question,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.1,
    )

    # 顯示答案
    answer = response.choices[0].text.strip()
    answer_list = answer.split("\n\n")
    return answer_list


def read_ask_gpt4(input_string):
    # 設定 API 金鑰
    openai.api_key = "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"

    # 定義問題
    question = input_string

    # 設定 GPT-3 模型的引擎 ID
    model_engine = "gpt-3.5-turbo"

    # 呼叫 OpenAI API 並取得答案
    response = openai.Completion.create(
        engine=model_engine,
        prompt=question,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.1,
    )

    # 顯示答案
    answer = response.choices[0].text.strip()
    answer_list = answer.split("\n\n")
    return answer_list


def check_issue(input_issue_name):
    try:
        temp_str = f"\"{input_issue_name}\"是否會導致網站不安全?請用是或否回答"
        temp_result = read_ask_gpt3(temp_str)[0]
        if temp_result[0] == "是":
            return True
        else:
            return False
    except Exception as ex:
        print('Exception:' + str(ex))
        return False


def read_cvss(input_issue_name):
    # init
    temp_result = 0
    try:
        temp_str = f"多數情況下,\"{input_issue_name}\"的 CVSSv3 分數最可能是多少? 請回答我數字"
        temp_result = read_ask_gpt3(temp_str)[0]
    except Exception as ex:
        print('Exception:' + str(ex))
    return temp_result


def read_owasp(input_issue_name):
    # init
    temp_result = ""
    temp_str = f"依序列出2017年 OWASP TOP 10的項目名稱與編號, {input_issue_name}問題屬於哪一種,編號是多少?"
    temp_answer = read_ask_gpt3(temp_str)
    temp_str = temp_answer[-1]
    if "A1" in temp_str:
        temp_result = "A03"
    elif "A2" in temp_str:
        temp_result = "A07"
    elif "A3" in temp_str:
        temp_result = "A02"
    elif "A4" in temp_str:
        temp_result = "A05"
    elif "A5" in temp_str:
        temp_result = "A01"
    elif "A6" in temp_str:
        temp_result = "A05"
    elif "A7" in temp_str:
        temp_result = "A03"
    elif "A8" in temp_str:
        temp_result = "A08"
    elif "A9" in temp_str:
        temp_result = "A06"
    elif "A10" in temp_str:
        temp_result = "A09"
    return temp_result


if __name__ == '__main__':
    test_issue_name = "Server Leaks Information via \"X-Powered-By\" HTTP Response Header Field(s)"
    print(test_issue_name)
    if check_issue(test_issue_name):
        print("CVSSv3: " + str(read_cvss(test_issue_name)))
        print("OWASP Top 10(2021): " + str(read_owasp(test_issue_name)))
    else:
        print("CVSSv3: 0")
        print("OWASP Top 10(2021): n/a")
