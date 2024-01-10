import openai
openai.api_key = "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"

model_engine = "text-davinci-001" # 設定模型引擎

while True:
    const = str(input())
    completions = openai.Completion.create(engine=model_engine, prompt=const, max_tokens=1024)
    message = completions.choices[0].text # 獲取模型回應
    print(message)
