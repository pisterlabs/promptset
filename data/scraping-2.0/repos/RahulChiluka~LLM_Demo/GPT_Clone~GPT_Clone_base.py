import chainlit as cl
import openai

openai.api_key = "ADD_YOUR_API_KEY_HERE"

def get_gpt_output(user_message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"you are a helpful assistant."},
            {"role":"user","content": user_message}
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response

@cl.on_message
async def main(message : str):
    await cl.Message(content = f"{get_gpt_output(message)['choices'][0]['message']['content']}",).send()
