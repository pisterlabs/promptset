import openai

user_message = input("質問したい内容を入力してください：")

res = openai.ChatCompletion.create(
	model = "gpt-3.5-turbo",
	messages = [
		{"role": "system", "content": "You are a helpful assistant."},
		{"role": "user", "content": user_message},
	]
)

res_content = res["choices"][0]["message"]["content"]

print(res_content)
