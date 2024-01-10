# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import sys
import streamlit as st

sys.path.append("...")

openai.api_key = st.secrets["open_ai_api_key"]
t = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant, who helps me creating text to image prompts. I want to sell t-shirt designs on the Merch by Amazon platform. Merch by Amazon is an on-demand t-shirt printing service. It allows sellers to create and list t-shirt designs on Amazon for free"},
        {"role": "user", "content": "In the following i will provide you multiple product descriptions from very successfull shirt designs which allready sell well. Your task is to understand the description and transform them to an text to image prompt at the end. If you understood answer with 'yes'."},
        {"role": "assistant", "content": "Yes, I understand. Please provide me with the product descriptions and I will help you transform them into text to image prompts."},
        {"role": "user", "content": "PewPew stands on outfit and is a fun saying. Great saying outfit with a rabbit and two pistols. Ideal for all rabbit lovers. Also at parties or celebrations with alcohol and festival with music. For lovers of sarcasm and joke. I will provide some more descriptions. If you understood answer with 'yes'."},
        {"role": "assistant", "content": "Yes, I understand. Please provide me with the additional product descriptions and I will help you transform them into text to image prompts."},
        {"role": "user", "content": "Pew Pew Madafakas women, Pew Pew Madafakas, Pew Pew Madafakas cat. Fun internet meme saying as a gift. Pew Pew Madafakas cat, Pew Pew Madafakas, Pew Pew Madafakas cat, ladies great gift idea for men and women. If you understood answer with 'yes'."},
        {"role": "assistant", "content": "Yes, I understand. Please provide me with the additional product descriptions and I will help you transform them into text to image prompts."},
        {"role": "user", "content": "Those descriptions are pretty long. I need you to transform them into a short concise text to image prompt, which is understandable for generative AIs such as Midjourney, stable diffusion or dalle2. The prompt should create a t-shirt which should sell well on the Merch by Amazon platform."},
        #{"role": "assistant", "content": "Yes, I understand. Please provide me with the additional product descriptions and I will help you transform them into text to image prompts."},
        #{"role": "user", "content": "What is a short text to image prompt which generates a perfect saleable t-shirt design based on my previous descriptions?"}
    ]
)
prompt = t["choices"][0]["message"].content
number_indicator = [str(i) for i in range(10)]
prompts = [p[3:] for p in prompt.split("\n") if p[0:1] in number_indicator]
print(prompt)
print(prompts)

completion = openai.Completion.create(deployment_id="deployment-name", prompt="Hello world")

test = 0