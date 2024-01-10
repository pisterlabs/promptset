"""
ref https://github.com/chatanywhere/GPT_API_free
"""

import openai

# openai.log = "debug"
openai.api_key = "sk-vsdk6LKfeVQm1H6DKRf4fpWZ6tMWnRVSWlZ1yGoXGzOnioSq"
openai.api_base = "https://api.chatanywhere.cn/v1"


# 非流式响应 completion = openai.ChatCompletion.create(models="gpt-3.5-turbo", messages=[{"role": "user", "content":
# "Hello world!"}]) print(completion.choices[0].message.content)

def gpt_35_api_stream(messages: list):
    """为提供的对话消息创建新的回答 (流式传输)

    Args:
        messages (list): 完整的对话消息

    Returns:
        tuple: (results, error_desc)
    """
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            stream=True,
        )
        completion = {'role': '', 'content': ''}
        for event in response:
            if event['choices'][0]['finish_reason'] == 'stop':
                print(f'收到的完成数据: {completion}')
                break
            for delta_k, delta_v in event['choices'][0]['delta'].items():
                # print(f'流响应数据: {delta_k} = {delta_v}')
                completion[delta_k] += delta_v
        messages.append(completion)  # 直接在传入参数 messages 中追加消息
        return True, ''
    except Exception as err:
        return False, f'OpenAI API 异常: {err}'


if __name__ == '__main__':
    message = [{'role': 'user', 'content': 'rewrite the following sentence: I’m loving it! We should learn from this app.'}, ]
    print(gpt_35_api_stream(message))
    print(message)
