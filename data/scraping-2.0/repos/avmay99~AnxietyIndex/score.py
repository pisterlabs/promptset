import os
import re
import openai

def get_average_score(keyword):
    numbers = re.findall(r'-?\d+', keyword)
    result = [int(number) for number in numbers]
    return sum(result) / len(result) * -1

def get_sentiment_score(text):
    api_key = os.environ.get('OPENAI_API_KEY')

    if api_key is None:
        raise Exception("OpenAI API密钥未设置")

    openai.api_key = api_key

    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=text,
        max_tokens=100,
        temperature=0.1,
        n=1,
        stop=None
    )

    sentiment_score = response.choices[0].text
    return sentiment_score

if __name__ == "__main__":
    # 示例文本
    input_text = '帮我分析如下语句的情感指数，-10代表极端负面， +10代表极端正面， 0代表中性，通过各种方式入行算作正面。每条语句输出一个数值\n"程序员26岁","程序员26岁表情包","程序员26岁,年薪三十万算低吗","程序员26岁坐在电脑面前受不了了"'

    # 获取情绪指数
    sentiment_score = get_sentiment_score(input_text)
    print(f"情绪指数: {sentiment_score}")