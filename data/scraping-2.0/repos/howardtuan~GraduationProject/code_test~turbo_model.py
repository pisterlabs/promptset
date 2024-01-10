import os
import openai
openai.api_key = 'api key'
while True:  
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "系統訊息，目前無用"},
            {"role": "assistant", "content": "此處填入機器人訊息"},
            {"role": "user", "content": input("You: ")}
        ]
    )
    print(completion.choices[0].message.content)

# 循環呼叫版本
# import openai

# openai.api_key = 'api key'
# messages = [
#     {"role": "system", "content": "系統訊息，目前無用"},
# ]

# while True:
#     user_input = input("You: ")
    
#     # 添加使用者輸入到對話歷史中
#     messages.append({"role": "user", "content": user_input})
    
#     # 使用完整的對話歷史呼叫模型
#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages
#     )
    
#     # 提取助手回應並輸出
#     assistant_response = completion.choices[0].message['content']
#     print("AI:", assistant_response)
    
#     # 添加助手回應到對話歷史中
#     messages.append({"role": "assistant", "content": assistant_response})
