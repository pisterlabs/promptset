from openai import OpenAI
import argparse
from product_info import get_product_json
import json
import config

client = OpenAI(api_key = config.key)
model = "gpt-3.5-turbo"

def chat_with_gpt(history, prompt):
	history += [{"role":"user", "content":prompt}]
	response = client.chat.completions.create(
		model=model,
		messages=history
	)
	output_text = response.choices[0].message.content
	history += [{"role":"assistant", "content":output_text}]
	return history, output_text

def send_first_message(msg):
	return "Product Information:" + json.dumps(msg) + "Please help me answer the following questions"

def start_conversation(barcode):
	print("ChatGPT Terminal Interaction\n")
	product_info = get_product_json(barcode)
	user_input = send_first_message(product_info)

	# conversation_history = []
	conversation_history = [{"role":"assistant", "content":user_input}]

	# print("You: " + user_input)
	# conversation_history, response = chat_with_gpt(conversation_history, user_input)
	# print("ChatGPT:", response)

	while True:
		user_input = input("You: ")
		if user_input.lower() == 'exit':
			print("Exiting chat.")
			break

		# Send user input to ChatGPT
		conversation_history, response = chat_with_gpt(conversation_history, user_input)

		# Display ChatGPT's response
		print("ChatGPT:", response)

if __name__ == "__main__":
	barcode = "038000222634"
	start_conversation(barcode)


# print(chat_with_gpt(''))
