import openai
openai.api_key = "your_api_key"

def generate_text(prompt):
	model_engine = "text-davinci-002"
	completion = openai.completion.create(engine=model_engine, prompt=prompt, max_tokens=2048)
	message = completion.choices[0].text
	return message
