from openai import OpenAI

import constants

client = OpenAI()
client.api_key = constants.api_key
def dalle(prompt,res,quality):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=res,
        quality=quality,
        n=1,
    )
    image_url = response.data[0].url
    return image_url
def chatGPT(creativity,model,systemPrompt,prompt):
    response = client.chat.completions.create(
        model= model,
        messages=[
            {
                "role": "system",
                "content": systemPrompt
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=creativity,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

def generateVisuals(prompt,model,creativity):
    return chatGPT(creativity,model,constants.PROMPT_VISUAL,prompt)

def generateSheet(prompt,model,creativity):
    return chatGPT(creativity,model,constants.PROMPT_SHEET,prompt)

def generateItemDescription(model,creativity):
    return chatGPT(creativity,model,"", constants.PROMPT_IDEAS)