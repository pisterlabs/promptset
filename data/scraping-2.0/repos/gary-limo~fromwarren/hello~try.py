import openai
openai.api_key = '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'  # supply your API key however you choose

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "RT @MikeTreglia: Our @nature_ny Cities team has an exciting opening for an Interdisciplinary Scientist!.is this text related to apple company. resond by yes or no only"}])

print(completion.choices[0].message.content)
