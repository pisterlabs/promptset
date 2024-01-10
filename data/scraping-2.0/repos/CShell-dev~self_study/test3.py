import openai

result=openai.ChatCompletion.create(
	model="gpt-3.5-turbo",
	max_tokens=256,
	temperature=0,
	messages=[
		{"role": "system", "content": "You are a helpful assistant."},
		{"role": "user", "content": "Create a blog article about large language model."}
        ]
)

print(result.choices[0].message.content)