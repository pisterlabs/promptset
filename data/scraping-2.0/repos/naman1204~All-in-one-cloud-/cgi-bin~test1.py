#!/usr/bin/python3
import cgi
import openai
print("content-type: text/html")
print()

form = cgi.FieldStorage()
prompt = form.getvalue("prompt")

mykey = "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"
openai.api_key = mykey

# Split the prompt into expert role and user question
expert_role, user_question = prompt.split("\n", 1)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=150,  # Increase the max_tokens to allow for longer responses
    temperature=0,
    messages=[
        {"role": "system", "content": "You are an expert language translator in all languages."},
        {"role": "user", "content": expert_role},
        {"role": "user", "content": user_question}
    ]
)
print(response.choices[0].message.content)

