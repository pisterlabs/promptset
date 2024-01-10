from flask import render_template, request
from openai import OpenAI

from app import app, openai_api_key

client = OpenAI(api_key=openai_api_key)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    user_input = request.form['text_input']

    response = client.images.generate(
        model="dall-e-3",  # Update with the appropriate DALL-E model
        prompt=user_input,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url

    return render_template('index.html', image_url=image_url, user_input=user_input)
