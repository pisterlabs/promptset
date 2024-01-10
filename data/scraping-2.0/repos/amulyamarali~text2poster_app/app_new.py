import streamlit as st
import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageFont, ImageDraw
import openai
import requests

openai.api_key = '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'

st.title("Text2Poster AI")
input_text = st.text_input("Enter the product name")

def generate_ad_tagline(prompt):
    prompt =  "Create a catchy short tagline for "+prompt
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can use a different engine if you prefer
        prompt=prompt,
        max_tokens=50,  # Adjust the number of tokens to control the length of the response
        temperature=0.7,  # Adjust the temperature for more or less randomness
        stop=None  # You can provide a list of stop tokens to control when to end the tagline
    )
    return response.choices[0].text.strip()

def gen_img_SD():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype = torch.float16)
    pipe = pipe.to("cuda")
    # pipe = pipe(input_text).image[0]
    image = pipe(input_text).images[0]
    image_editable = ImageDraw.Draw(image)
    title_font = ImageFont.truetype("/content/Poppins-Medium.ttf",50)
    tagline = generate_ad_tagline(input_text)
    image_editable.text((10,10), tagline,(237,230,211), font = title_font)
    st.write(tagline)
    st.image(image)
    # return image

button = st.button("Generate",on_click=gen_img_SD)
# st.image(gen_img_SD("Sports Shoe"))
