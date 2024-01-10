import openai

openai.api_key = "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"

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "who is the current president of nigeria "}])
print(completion.choices[0].message.content)
