from openai import OpenAI
import os

OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

response = client.chat.completions.create(
model="gpt-4-1106-preview",

messages=[
	{"role": "user", "content": "Who is Charlemagne?"}
	],
	stream=True
)

# Print out the response from the model.
for event in response:
    event_text = event.choices[0].delta
    answer = event_text.content

    # Test if answer is not a NoneType
    if answer is None:
        answer = ''
	
    print(answer, end='', flush=True)