import os
import openai

def get_gpt_response(model_name, user_content, system_content=''):
    openai.organization = os.environ['OPENAI_KUNLP']
    openai.api_key = os.environ['OPENAI_API_KEY_TIMELINE']

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    )

    return response['choices'][0]['message']['content']