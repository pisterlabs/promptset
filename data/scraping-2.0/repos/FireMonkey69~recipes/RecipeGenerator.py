import streamlit as st
import openai
import random
import requests
from PIL import Image

 

# Set up OpenAI API credentials
OPEN_API_KEY = st.secrets["OPEN_API_KEY"]
openai.api_key = OPEN_API_KEY

 


# Define function to generate recipe using ChatGPT
def generate_recipe(ingredients):
    prompt = f"I have {', '.join(ingredients)} in my pantry. Can you suggest a recipe?"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.7,
    )
    recipe = response.choices[0].text.strip()
    return recipe

 

def gen_image(txt):

 

    # Generate image using DALL-E
    response = requests.post(
           "https://api.openai.com/v1/images/generations",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer "+ OPEN_API_KEY,
            },
            json={
                "model": "image-alpha-001",
                "prompt": txt,
                "num_images": 1,
                "size": "1024x1024",
                "response_format": "url",
            },
        )
    image_url = response.json()["data"][0]["url"]

 

    # Display image in Streamlit app
    image = Image.open(requests.get(image_url, stream=True).raw)
    st.image(image, caption="Generated Image")

 

# Define Streamlit app
def app():
    st.title("Recipe Generator")
    st.write("Enter a list of ingredients and I'll suggest a recipe!")

 

    # Get user input of ingredients
    ingredients = st.text_input("Enter ingredients, separated by commas")

 

    # Generate recipe using ChatGPT
    if st.button("Generate Recipe"):
        recipe = generate_recipe(ingredients.split(","))
        st.write(recipe)
        gen_image(recipe)

 

if __name__ == '__main__':
    app()
