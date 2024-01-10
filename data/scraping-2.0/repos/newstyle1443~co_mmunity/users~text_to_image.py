import openai
from co_mmunity.settings import get_secret


openai.api_key = get_secret("API_KEY")

def text_to_image(user_keyword):
    gpt_prompt = []
    gpt_prompt.append({
        "role":"system",
        "content":"Translating Korean into English in detail."
    })
    gpt_prompt.append({
        "role":"system",
        "content":"Imagine the detail appearance of the input. Response shortly. Translating Korean into English in detail."
    })
    gpt_prompt.append({
        "role":"user",
        "content":user_keyword
    })
    
    prompt = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=gpt_prompt)
    prompt = prompt["choices"][0]["message"]["content"]
    
    result = openai.Image.create(prompt=prompt, size="512x512")
    return result['data'][0]['url']