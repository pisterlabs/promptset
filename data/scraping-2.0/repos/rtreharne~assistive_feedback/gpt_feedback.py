from config import OPENAI_API_KEY
from openai import OpenAI


def gpt_response(prompt, text):

	cost_per_token = 0.001/1000

	# Create the client
	client = OpenAI(
		api_key=OPENAI_API_KEY
	)

	response = client.chat.completions.create(
	model="gpt-3.5-turbo-1106",
	messages=[
		{
		"role": "system",
		"content": prompt
		},
		{
		"role": "user",
		"content": f"Give me some feedback for this reference list:\n{text}"
		}
	],
	temperature=0.1,
	max_tokens=1000,
	top_p=1,
	frequency_penalty=0,
	presence_penalty=0
	)

	return response.choices[0].message.content, response.usage.total_tokens*cost_per_token

    
if __name__ == "__main__":
    # Get prompt text from prompt.txt
	with open('prompt.txt', 'r', encoding='utf-8') as f:
		prompt = f.read()

	# Get text from example_references.txt
	with open('example_references.txt', 'r', encoding='utf-8') as f:
		text = f.read()

	text, cost = gpt_response(prompt, text)


	# Save text to .txt file
	with open('response.txt', 'w') as f:
		f.write(text)

	print(text, cost)


