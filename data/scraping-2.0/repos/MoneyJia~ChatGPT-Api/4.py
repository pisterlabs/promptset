import openai
import os

os.environ["all_proxy"] = 'http://127.0.0.1:10792' #对应代理地址或科学上

openai.api_key = "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"

complection = openai.ChatCompletion.create(
    model="text-davinci-003",
    prompt="test Completions api",
    max_tokens= 7,
    temperature= 0
)

print(complection.choices[0].message)

response = openai.Completion.create(
    engine="text-davinci-003",  # 使用适当的引擎（模型）
    prompt="Translate the following English text to French: '{}'",
    max_tokens=50  # 生成文本的最大长度
)

print(response.choices[0].text)
