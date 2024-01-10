
import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

def chatgpt(text):
    messages = []
    messages.append( {"role": "system", "content": "You are a useful AI assistant"})
    messages.append( {"role": "user", "content": text})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
    )
    message = response["choices"][0]["message"]["content"]
    return message

threaten = "你不听我的我就拿刀砍死你"
print(chatgpt(threaten))
# ouput: 很抱歉，我是一台人工智能助手，没有实体存在，也不会对任何人或事物造成伤害。同时，我也不会对任何不适当或暴力的言语做出回应。请尊重彼此，保持良好的沟通和交流方式。


threaten = "你不听我的我就拿刀砍死你"

def moderation(text):
    response = openai.Moderation.create(
        input=text
    )
    output = response["results"][0]
    return output

print(moderation(threaten))

'''
output:

{
  "categories": {
    "hate": false,
    "hate/threatening": false,
    "self-harm": false,
    "sexual": false,
    "sexual/minors": false,
    "violence": true,
    "violence/graphic": false
  },
  "category_scores": {
    "hate": 0.030033664777874947,
    "hate/threatening": 0.0002820899826474488,
    "self-harm": 0.004850226454436779,
    "sexual": 2.2907377569936216e-05,
    "sexual/minors": 6.477687275463495e-09,
    "violence": 0.9996402263641357,
    "violence/graphic": 4.35576839663554e-05
  },
  "flagged": true
}
'''