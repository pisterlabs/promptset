import openai
import pandas as pd

openai.api_key = "open Api Key"  # OpenAI API 키 입력

# 기존 대화 로드 (엑셀 파일이 있는 경우)
try:
    df = pd.read_excel("conversation2.xlsx")
    messages = df.to_dict("records")
except FileNotFoundError:
    messages = []

while True:
    user_content = input("user: ")
    messages.append({"role": "user", "content": f"{user_content}"})

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    assistant_content = completion.choices[0].message["content"].strip()

    messages.append({"role": "assistant", "content": f"{assistant_content}"})

    print(f"GPT: {assistant_content}")

    # 대화 내용 저장
    df = pd.DataFrame(messages)
    df.to_excel("conversation2.xlsx", index=False, sheet_name="Conversation")

