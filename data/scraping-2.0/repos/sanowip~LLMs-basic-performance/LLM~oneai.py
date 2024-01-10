import os
import openai


# OPENAI_API_KEY是自己设定的环境变量名
openai.api_key = os.getenv("OPENAI_API_KEY")
# 明文
openai.api_key = ""
openai.api_base="https://wechatchat.tech/v1/chat"

prompt = "鲁迅打周树人"
response = openai.Completion.create(
    
    model= "gpt-4-1106-preview",
    temperature=0.5,
    max_tokens=1000,
    n=1,    
    stop=None,
    timeout=20,
    messages=[  
        {"role": "user", "content": prompt},
    ],
)

print(response.choices[0].message['content'])
