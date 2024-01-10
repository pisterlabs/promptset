
from openai import OpenAI


# const apiKey = process.env.OPENAI_API_KEY!;
# const OrganizationID = process.env.OPENAI_API_ORG_ID;

client = OpenAI()
client.api_key = "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"
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)