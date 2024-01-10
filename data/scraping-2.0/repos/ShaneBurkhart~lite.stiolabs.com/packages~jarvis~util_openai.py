import openai 

openai.api_key = os.environ.get("OPENAI_KEY")

def generate_response_smart(prompt):
	response = openai.Completion.create(
		model="text-davinci-003",
		prompt=prompt,
		temperature=0,
		max_tokens=256,
		top_p=1,
		frequency_penalty=0,
		presence_penalty=0
	)
	# response = openai.Completion.create(
	# 	model="davinci",
	# 	prompt=prompt,
	# 	temperature=0
	# )

	# print(response)
	return response.choices[0].text.strip()
