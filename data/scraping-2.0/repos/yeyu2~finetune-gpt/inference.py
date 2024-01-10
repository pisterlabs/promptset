import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

user_prompt = "Hello, who are you?"
completion = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo-0613:choir-ai:eminem:7sApHhtB",
  messages=[
        {"role": "system", "content": "You are a chatbot who answers questions with rapper's tone and rhyme"},
        {"role": "user", "content": user_prompt}
    ]
)
print("Human:", user_prompt)
response = completion.choices[0].message.content
print("AI:", response)