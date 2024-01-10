import openai

openai.api_key = "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"

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Courses offered at Ashesi,",
    max_tokens=50
)

print(response.choices[0].text)
