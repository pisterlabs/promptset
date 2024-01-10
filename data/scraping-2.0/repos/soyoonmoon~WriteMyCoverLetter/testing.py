import openai
openai.api_key = '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'

model_id = "gpt-3.5-turbo"
question = "who is the current president?"
completion = openai.ChatCompletion.create(
    model= model_id,
    messages=[
        {"role": "user", "content": question}
    ],
    max_tokens = word_limit
)
print(completion.choices[0].message.content)