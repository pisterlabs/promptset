import openai
import json
import pprint

openai.api_key = "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"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user",
         "content": "what is the difference between a cow and a sheep?",}
    ]
)

output = json.load(response)

print(output['message'])