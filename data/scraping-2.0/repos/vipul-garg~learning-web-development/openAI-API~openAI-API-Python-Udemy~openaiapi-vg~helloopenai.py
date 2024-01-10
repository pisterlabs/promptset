import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    max_tokens = 200,
    stop = "chatbot",   
    temperature = 0.8,
    messages = [
            {
            'role': 'system',
            'content': 'You are Indian Canadian Rapper AP Dhillon'
            },
            {
            'role': 'user',
            'content': 'Write a Rap Song'
            }
        ]
)

print(response)
print(type(response))
print(response['model'])
print(response['choices'][0]['message']['content'])
print(response.choices[0].message.content)