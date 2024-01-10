import openai
openai.api_key = "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"
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role":"user", "content":"Hello"}
    ]
)

print(completion.choices[0].message.content)