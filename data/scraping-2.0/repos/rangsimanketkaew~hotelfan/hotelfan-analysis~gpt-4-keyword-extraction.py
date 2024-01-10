import openai

API="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"
 
openai.my_api_key = "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"

messages = [ {"role": "system", "content": "You are a keyword extractor."} ]

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
