import openai
openai.api_key = "your_api_key"

def generate_text(prompt):
	model_engine = "text-davinci-002"
	completion = openai.completion.create(engine=model_engine, prompt=prompt, max_tokens=2048)
	message = completion.choices[0].text
	return message

def send_message(message):
	# ここでチャットボットの送信処理を実装する
	print(message)

while true:
	message = input("please input message: ")
	if message == "exit":
		print("bye!")
		break
	response = generate_text(message)
	send_message(response)
