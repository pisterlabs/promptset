import os
import openai
import email_config
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = email_config.OPENAI_API_KEY
# openai.api_key = 'sk-2ZLc2m4dVc7h4h3tH8ZnqK1m5V7mJZ'


def call_gpt_api(content):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are my assistant, an expert in all sort of matters"},
        {"role": "user", "content": content}
    ]
    )

    # print(completion.choices[0].message)
    return completion.choices[0].message

