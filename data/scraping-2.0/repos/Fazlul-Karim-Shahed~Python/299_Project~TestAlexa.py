import openai

openai.api_key = "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"

response = openai.Completion.create(
# model = "gpt-3.5-turbo",
# messages = [ {"role": "user", "content": 'hello'} ]
engine='davinci',
prompt='hello'
)

print(response.choices[0])