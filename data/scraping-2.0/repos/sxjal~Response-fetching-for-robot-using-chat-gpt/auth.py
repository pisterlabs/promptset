import os
import openai
openai.organization = "org-nXWvlzifxQRNmd6qZTI8ijaR"
openai.api_key = "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"
# print(openai.Model.list())


messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]

while True:
	message = input("User : ")
	if message:
		messages.append(
			{"role": "user", "content": message},
		)
		chat = openai.ChatCompletion.create(
			model="gpt-3.5-turbo", messages=messages
		)
	
	reply = chat.choices[0].message.content
	print(f"ChatGPT: {reply}")
	messages.append({"role": "assistant", "content": reply})
