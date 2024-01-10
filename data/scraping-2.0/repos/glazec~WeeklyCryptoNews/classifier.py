import openai
import os


def classifier(text):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    with open('prompt.txt', 'r') as file:
        prompt = file.read()
    prompt += '\n'+'News\n'+text + '\nTag:'
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    # print(text, response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']


if __name__ == 'main':
    classifier('数据：过去一周 Circle USDC 流通量减少 4 亿美元')
