import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def ai(prompt):
	return openai.Completion.create(
	model="text-davinci-003",
	prompt=prompt,
	temperature=0.9,
	max_tokens=150,
	top_p=1,
	frequency_penalty=0.0,
	presence_penalty=0.6,
	stop=[" Human:", " AI:"]
	)
prompt='-'
while prompt.strip():
	if not prompt=='-':
		response = ai(prompt)
		print('AI:',response['choices'][0]['text'].strip())
	prompt=input('Human: ')
