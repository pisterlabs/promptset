import openai
from interactions import SlashContext
from load_data import *

openai.api_key = load_config("OpenAI")


async def chat(context: SlashContext, prompt):
    system_prompt = ''

    with open('Data/askprompt.txt', 'r') as f:
        system_prompt = f.read()

    system_prompt = system_prompt.replace('+user', context.author.mention)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9
    )

    return response['choices'][0]['message']['content']


async def generate_text(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return response['choices'][0]['message']['content']
