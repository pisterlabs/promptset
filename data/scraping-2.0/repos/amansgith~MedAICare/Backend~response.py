import os
import openai
import gradio as gr
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import io

# Replace 'YOUR_API_KEY' with your actual API key from OpenAI
openai.api_key = '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'

app = Flask(__name__)

def preprocess_image(img):
    # # Resize the image to a fixed size (e.g., 224x224)
    # img = img.resize((224, 224))
    
    # # Convert to NumPy array
    # img_array = np.array(img)
    
    # # Normalize pixel values to the range [0, 1]
    # img_array = img_array / 255.0
    
    # return img_array
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img_arr = np.expand_dims(img, 0)
    return img

def chat_with_gpt(input_text):
    response = openai.Completion.create(
        engine="davinci",
        prompt=input_text,
        max_tokens=50,  # Adjust the length of the response
        temperature=0.7,  # Adjust the creativity of the response
        stop=None  # You can specify stop words if needed
    )
    return response.choices[0].text.strip()

iface = gr.Interface(
    fn=chat_with_gpt,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.Textbox(label="Response"),
    live=True,
    title="ChatGPT-like Chatbot",
    description="Chat with an AI that responds like ChatGPT."
)

@app.route("/", methods=["GET", "POST"])
def classify_image():
    prescription = None

    if request.method == "POST":
        # Get the uploaded image
        uploaded_image = request.files["image"].read()
        img = Image.open(io.BytesIO(uploaded_image))

        # Preprocess the image (resize, normalize, etc.)
        img = preprocess_image(img)

        # Use the trained model to make a prediction (you can add your model prediction logic here)
        # For this example, we're using the ChatGPT-like chatbot
        input_text = request.form["text"]
        prescription = chat_with_gpt(input_text)

    return render_template("result.html", prescription=prescription)

if __name__ == "__main__":
    app.run(debug=True)
