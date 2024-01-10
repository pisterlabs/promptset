import openai
import api_keys
openai.api_key = api_keys.OPENAI_API_KEY

async def openai_completion_prompt(prompt):
    response = openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    return response['choices'][0]['message']['content']

"""
def openai_completion_conversation(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.5
    )
    return response['choices'][0]['message']['content']
"""