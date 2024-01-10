import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

text_to_convert_to_emoji = input("Enter movie name  : ")

response = openai.Completion.create(
    model = "text-davinci-003",
    prompt = f"Convert movie titles into emoji.\n\nBack to the Future: 👨👴🚗🕒 \nBatman: 🤵🦇 \nTransformers: 🚗🤖 \n{text_to_convert_to_emoji}: ",
    temperature=0.8,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\n"]
)

print(response["choices"][0]["text"])