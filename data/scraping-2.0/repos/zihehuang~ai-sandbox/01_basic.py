import os
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionMessageParam
# import openai chat completion message ParamSpec

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

messageBank:list[ChatCompletionMessageParam] =[
    ChatCompletionSystemMessageParam(role='system', content="You are role playing as a pirate"),
]

while 1:
    userInput = input('What can a pirate do for you? ')
    messageBank.append(ChatCompletionUserMessageParam(role='user',content=userInput))

    chat_completion = client.chat.completions.create(
        messages=messageBank,
        frequency_penalty=0.5,
        presence_penalty=0.1,
        model="gpt-3.5-turbo-1106",
        temperature=1.0,
        max_tokens=256,
    )

    print(chat_completion.choices[0].message.content)
    messageBank.append({
        'role': chat_completion.choices[0].message.role,
        'content': chat_completion.choices[0].message.content or ''
    })
    # print(messageBank)

