import openai
import os
from dotenv import load_dotenv

def main():
	load_dotenv()
	openai.api_key = os.getenv("OPENAI_API_KEY")

	test_messages = []
	test_messages.append({"role": "user", "content": "Hi, what are you here to help with?"})
	test_messages.append({"role": "user", "content": "Can you give me a hello world example of lwfm using my local machine as a Site?"})

	response = openai.ChatCompletion.create(model=fine_tuned_model_id, message=test_messages, temperature=0, max_tokens=500)

	print(test_messages[0]["content"])
	print(response[choices][0]["message"]["content"])
	print(test_messages[1]["content"])
	print(response[choices][1]["message"]["content"])

	response = openai.ChatCompletion.create(model="gpt-3.5-turbo", message=test_messages, temperature=0, max_tokens=500)

	print(test_messages[0]["content"])
	print(response[choices][0]["message"]["content"])
	print(test_messages[1]["content"])
	print(response[choices][1]["message"]["content"])

