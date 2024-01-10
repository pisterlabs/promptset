import gradio as gr
import openai
from keras.models import load_model
import cv2
import numpy as np

# Class labels for face prediction
class_labels = {
    0: 'Anushka_Sharma',
    1: 'Bill_Gates',
    2: 'Dalai_Lama',
    3: 'Narendra_Modi',
    4: 'Sundar_Pichai'
}

# Function to preprocess the input image
def preprocess_image(img):
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize the image to match the expected input shape of the model
    img = cv2.resize(img, (224, 224))
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0
    return img

# Function to predict a name and generate a description from an image
def predict_and_generate_description(img):
    # Load the model
    model_path = r'C:\Users\anoop\OneDrive\Desktop\vs_code\api\keras_model (2).h5'  # Replace with the path to your model
    model = load_model(model_path)
    
    # Preprocess the input image
    img = preprocess_image(img)
    
    # Predict with the model
    prediction = model.predict(np.expand_dims(img, axis=0))
    
    # Get the index of the predicted class
    predicted_class_index = np.argmax(prediction)
    
    # Get the predicted class label
    predicted_class = class_labels.get(predicted_class_index, "Unknown")
    
    # Set up your OpenAI API key
    openai.api_key = "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"  # Replace with your OpenAI API key
    
    # Use GPT-3.5 to generate a description based on the name prediction
    prompt = f"Generate a description for {predicted_class}."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=500  # Adjust the desired length of the description
    )
    
    description = response.choices[0].text
    
    return f"Name: {predicted_class}\nDescription: {description}"

# Create a Gradio interface
iface = gr.Interface(fn=predict_and_generate_description, inputs="image", outputs="text")

# Launch the interface
iface.launch()
