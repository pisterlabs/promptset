import os

from openai import OpenAI
# token
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")
chinese = "宛如宿命一般，如今他再次被卷入“命运转盘”的争夺，而纳粹的残余势力也卷土重来，觊觎着这件宝物"
tokens = encoding.encode(chinese)
print(tokens)
print(len(tokens))


api_key = os.environ.get("OPEN_AI_API_KEY")
print(api_key)
                     
client = OpenAI(
    api_key = api_key
)

def test():
    response = chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "你是一个AI助理"},   
            {"role": "user", "content": "你好！你叫什么名字？"}
        ],
        temperature = 0.9, # (0~2), 越小越稳定
        max_tokens = 200,
        model="gpt-4",
        # n = 3, # 回复个数
    )
    for choice in response.choices:
        print(choice.message.content)

# test()

