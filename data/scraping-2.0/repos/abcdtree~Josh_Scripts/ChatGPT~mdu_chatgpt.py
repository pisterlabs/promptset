import os
import subprocess
import argparse
import openai

openai.api_key = "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"
model_engine = "text-davinci-003"

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("CHAT", help="message you want to send to CHATGPT")
	args = parser.parse_args()

	prompt = args.CHAT

	# Generate a response
	completion = openai.Completion.create(
		engine=model_engine,
		prompt=prompt,
		max_tokens=1024,
		n=1,
		stop=None,
		temperature=0.5,
	)

	response = completion.choices[0].text
	print("ChatGPT:\n",response)

if __name__ == "__main__":
	main()
