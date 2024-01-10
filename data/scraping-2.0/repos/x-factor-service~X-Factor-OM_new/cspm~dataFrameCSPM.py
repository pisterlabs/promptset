import openai
openai.api_key = "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"

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="BTS의 나이를 알려줘 2023년 6월 1일 기준으로",
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

#print(response['choices'][0]['text'])
