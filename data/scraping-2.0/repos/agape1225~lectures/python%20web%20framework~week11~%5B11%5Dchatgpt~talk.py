import os
import openai

#api_key = os.environ["api_key"]
openai.api_key = "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"


while True :
    prompt = input()
    if prompt == "." : break
    print("Me : " + prompt)    

    messages = []
    messages.append({"role": "user", "content": prompt})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  messages=messages)

    print(completion)
    res= completion.choices[0].message['content']
    
    print("GPT : " +res)