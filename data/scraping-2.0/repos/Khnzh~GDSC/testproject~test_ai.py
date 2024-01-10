# import openai
# openai.api_key = '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'
# def chat_with_gpt(prompt):
#     user_input = f'User: {prompt}\nAI:'
#     response = openai.Completion.create(
#         engine="gpt-3.5-turbo-instruct",
#         prompt=user_input,
#         max_tokens=150,
#         temperature=0.7,
#     )
#     ai_message = response['choices'][0]['text'].strip()
#     print(f'AI: {ai_message}')
# while True:
#     user_prompt = input("You: ")
#     if user_prompt.lower() == 'exit':
#         print("Goodbye!")
#         break
#     chat_with_gpt(user_prompt)
import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPEN_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)