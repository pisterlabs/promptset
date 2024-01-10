from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="", # ここにモデルIDを入力
    messages=[
        {"role": "system", "content": "あなたは天才機械学習エンジニアです. "},
        {"role": "user", "content": "LangChainとは何ですか？"}
    ]
)

print(response.choices[0].message)