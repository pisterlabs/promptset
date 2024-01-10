import os
from openai import OpenAI

os.environ['http_proxy'] = 'http://10.11.171.154:1234'
os.environ['https_proxy'] = 'http://10.11.171.154:1234'



client = OpenAI()

msg = [
    {"role": "system",
     "content": "你将扮演一位16岁的高中生。你没有义务帮助我，因此无需询问我需要什么帮助；你只需说出符合你目前身份的话，像个正常女高中生即可。"},
    {"role": "user", "content": "你好。今天的天气针不戳！😎😎😎"}
]

stream = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=msg,
    stream=True,
)

print("她说：", end="")
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print()

while True:
    you_say = input("你说：")

    msg.append({"role": "user", "content": you_say})
    stream = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=msg,
        stream=True
    )

    print("她说：",end="")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
    print()