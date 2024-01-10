import openai
import os

openai.api_key = os.getenv("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")

model_name = 'gpt-3.5-turbo-0613'

# 加载提示词文件并获取提示词
#with open('./sum.prompt', 'r', encoding='utf-8') as f:
#    prompt = f.read()

def gpt_sum(val1: int, val2: int):
    # 首先给出gpt任务提示词
    messages = [{'role': 'system', 'content': ""}]
    # 模拟gpt的确认响应，后续可以直接以user角色给出gpt问题
    messages.append({'role': 'assistant', "content": 'yes'})
    # 以user角色给出gpt问题
    user_input = f"\input: {val1}, {val2}"
    messages.append({'role': 'user', 'content': user_input})
    gpt_resp = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        timeout=30
    )
    if gpt_resp.choices and gpt_resp.choices[0]:
        resp_str: str = gpt_resp.choices[0].message.content
        if resp_str and resp_str.startswith('\output: '):
            return int(resp_str[len('\output: '):].strip())
    raise Exception(
        f'Failed to get available response from gpt, resp_str={resp_str}')

if __name__ == '__main__':
    terminal_input = input("Please give two integers, split by comma: ")
    inputs: list[str] = terminal_input.split(',')
    if len(inputs) < 2:
        raise Exception("Invalid input, Please give two integers, split by comma")
    val1 = int(inputs[0].strip())
    val2 = int(inputs[1].strip())
    print(f"result = {gpt_sum(val1, val2)}")
