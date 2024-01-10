import json
from random import random
from time import sleep



def step_2():
    import openai
    openai.api_key="sk-BcoWEcTXXX2vly"
    # model_engien = "chatgpt"
    model_engien = "text-davinci-003"

    with open('D:/phd5/KR/natural_questions_answers.jsonl', 'a', encoding='utf-8') as fw:
        with open('D:/phd5/KR/natural_questions.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                id = json_obj['id']
                if id <= 1771:
                    continue
                if random() > 0.5:
                    sleep(2)
                length = json_obj['length']
                question = json_obj['question']
                completions = openai.Completion.create(engine=model_engien, prompt=question, max_tokens=1024, n=1, stop=None, temperature=0.5)
                message = completions.choices[0].text
                print(message)
                json.dump({"id": id, "length": length, "prompt": question, "answer": message}, fw)
                fw.write('\n')


def step_22():
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    import time

    # 创建一个浏览器实例
    driver = webdriver.Chrome()

    # 访问目标网站
    driver.get("https://chat.openai.com/chat")

    # 找到输入框并输入文本
    input_box = driver.find_element_by_xpath("//textarea[@class='queryInput']")
    input_box.send_keys("Hello, ChatGPT!" + Keys.RETURN)

    # 等待网页加载完成
    time.sleep(10)

    # 获取服务器返回值
    result = driver.page_source

    # 关闭浏览器实例
    driver.quit()


if __name__ == '__main__':
    step_1()
    step_2()
    # step_22()