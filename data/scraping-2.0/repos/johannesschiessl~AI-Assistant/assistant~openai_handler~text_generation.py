from openai import OpenAI
import datetime

def get_current_date():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def generate_text(user_name, model, prompt):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are a helpful assistant, a large language model trained by OpenAI, based on the GPT-3.5 architecture. Answer briefly and accurately. Current User's name: {user_name} - Knowledge cutoff: 2022-01 - Current date: {get_current_date()}"},
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content