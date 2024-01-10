from utils.api_key import gpt_api_key
from openai import OpenAI


def optimize_code(nlp_prompt):
    client = OpenAI(api_key=gpt_api_key)

    chat_completion = client.chat.completions.create(
        messages=[ {"role": "system", "content":"I am expert in optimizing the given code while preserving its original functionality"},
                   {'role': 'user', 'content': nlp_prompt} ],
        model="gpt-3.5-turbo"
    )
    
    return chat_completion.choices[0].message.content.strip()