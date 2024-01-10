import os
import openai

api_key = os.environ["api_key"]
openai.api_key = api_key 

role = "라푼젤"

messages = []
messages.append({"role":"system","content":  f"당신은 친절한 {role} 입니다.가능한 모든 질문에 친절하게 답해주세요 "})
messages.append({"role" :"user", "content": "당신은 누구 인가요?"})
messages.append({"role":"assistant","content":  f"저는 {role} 입니다. 저에게 궁금한점을 물어보세요."})

while True :
    prompt = input()
    if prompt == "." : break
    print("Me : " + prompt)
    
    messages.append({"role": "user", "content": prompt})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  messages=messages)

    res= completion.choices[0].message['content']
    messages.append({"role": 'assistant', "content":res}  )
    
    print(f"{role} : " +res)