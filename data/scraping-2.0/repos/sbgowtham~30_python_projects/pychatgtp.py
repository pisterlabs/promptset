import openai
openai.api_key = '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'
messages = [ {"role": "system", "content":
			"You are a philosopher ."} ]
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
