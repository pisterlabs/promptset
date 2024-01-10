import openai

openai.api_key = "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"

def hexcode(input):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="The CSS code for a color like :" + input + "\n\nbackground-color: #",
        temperature=0,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=[";"]
    )
    return response.choices[0].text

def sentences(input):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=input + "\n\nYou can be decribed in six sentences below:",
        temperature=0,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.text
