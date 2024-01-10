import openai

def make_short_text(text):
    messages = []
    messages.append({"role": "system", "content": "你是一个用来将文本改写得短的AI助手，用户输入一段文本，你给出一段意思相同，但是短小精悍的结果"})
    messages.append({"role": "user", "content": text})
    response = openai.ChatCompletion.create(
        model="gpt-4", #"gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        max_tokens=2048,
        presence_penalty=0,
        frequency_penalty=2,
        n=3,
    )
    return response

long_text ="""
在这个快节奏的现代社会中，我们每个人都面临着各种各样的挑战和困难。
在这些挑战和困难中，有些是由外部因素引起的，例如经济萧条、全球变暖和自然灾害等。
还有一些是由内部因素引起的，例如情感问题、健康问题和自我怀疑等。
面对这些挑战和困难，我们需要采取积极的态度和行动来克服它们。
这意味着我们必须具备坚韧不拔的意志和创造性思维，以及寻求外部支持的能力。
只有这样，我们才能真正地实现自己的潜力并取得成功。
"""
short_version = make_short_text(long_text)

index = 1

for choice in short_version["choices"]:
    print(f"version {index}: " + choice["message"]["content"])
    index += 1
