import os
from openai import OpenAI

client = OpenAI()

messages = [
    [
      {"role": "system", "content": "You are a helpful medical prescription chatbot."},
      {"role": "user", "content": "What is the use of following drug: 'Admenta 5mg Tablet 10'SAdmenta 10mg Tablet 10'S.'?"} # 3
    ],
    [
      {"role": "system", "content": "You are a helpful medical prescription chatbot."},
      {"role": "user", "content": "What is the use of following drug: 'Modapro 100mg Tablet 10'SModapro 200mg Tablet 10'S.'?"} #1
    ],
    [
      {"role": "system", "content": "You are a helpful medical prescription chatbot."},
      {"role": "user", "content": "What is the use of following drug: 'Waklert 150mg Tablet 5'SWaklert 150mg Tablet 10'SWaklert 250mg Tablet 5'SWaklert 50mg Tablet 10'SWaklert 100mg Tablet 10.'?"} # 1
    ],
    [
      {"role": "system", "content": "You are a helpful medical prescription chatbot."},
      {"role": "user", "content": "What is the use of following drug: 'Ebal 20mg Tablet 10'SEbal 10mg Tablet 10'S'?"} #2
    ],
    [
      {"role": "system", "content": "You are a helpful medical prescription chatbot."},
      {"role": "user", "content": "What is the use of following drug: 'Lasma LC Kid Tablet 10'S.'?"} #2
    ],
    [
      {"role": "system", "content": "You are a helpful medical prescription chatbot."},
      {"role": "user", "content": "What is the use of following drug: 'N Citi Plus 500\/800mg Tablet 10'S.'?"} #3
    ],
    [
      {"role": "system", "content": "You are a helpful medical prescription chatbot."},
      {"role": "user", "content": "What is the use of following drug: 'Tomoxetin 18mg Capsule 10'S.'?"} #1
    ],
    [
      {"role": "system", "content": "You are a helpful medical prescription chatbot."},
      {"role": "user", "content": "What is the use of following drug: 'Furazole M Tablet 10'S.'?"} #4
    ],
    [
      {"role": "system", "content": "You are a helpful medical prescription chatbot."},
      {"role": "user", "content": "What is the use of following drug: 'Metrogyl Compound Plus Tablet 10'SMetrogyl Compound Tablet 15'S.'?"} #4
    ],
  ]

for message in messages:
  response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-1106:personal::8Y67cEkf",
    messages=message,
    #prompt=prompt,
    temperature=0,
    max_tokens=100,
  )
  print(response.choices[0].message.content)
  print('#'*50)