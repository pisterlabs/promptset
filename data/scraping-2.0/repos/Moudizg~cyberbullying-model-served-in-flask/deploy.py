from flask import Flask, request, jsonify, render_template, redirect, url_for
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import json


import openai


# load the api key
api_key = ""
openai.api_key = api_key
def predict_class_gpt3(text):
    # print join(list(label_mapping.keys()))
    print(list(label_mapping.keys()))
    messages=[
        {"role": "system", "content": "You are a classifier model. Your task is to classify a text into one of the following categories, only reply with the category: " + ', '.join(list(label_mapping.keys()))},
        {"role": "user", "content": f"The text is: '{text}'"},
        {"role": "system", "content": "The predicted class is: "}
    ]

    # Make API request to GPT-3
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=20,
    )

    # Process GPT-3 response
    gpt_response = response['choices'][0]['message']['content'].strip()

    # Get the first word only
    first_word = gpt_response.split()[0]

    return first_word


app = Flask(__name__)

model_name = 'cyberbullying_collab'
tokenizer = DistilBertTokenizer.from_pretrained('cyberbullying_collab')

model = TFDistilBertForSequenceClassification.from_pretrained('cyberbullying_collab')

with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

def predict_class(text, model, tokenizer):
    inputs = tokenizer.encode(text, return_tensors='tf')
    predictions = model(inputs)[0]
    predicted_class_idx = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_class = label_mapping_inverse[predicted_class_idx]
    return predicted_class

label_mapping_inverse = {v: k for k, v in label_mapping.items()}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['text']

    # Prediction from our model
    predicted_class = predict_class(data, model, tokenizer)

    # Prediction from GPT-3
    gpt3_prediction = predict_class_gpt3(data)

    # Return both predictions
    return jsonify({'model_prediction': predicted_class, 'gpt3_prediction': gpt3_prediction})



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cast_vote', methods=['POST'])
def cast_vote():
    data = request.get_json()
    text = data.get('text')
    model_prediction = data.get('model_prediction')
    gpt3_prediction = data.get('gpt3_prediction')
    vote = data.get('vote')
    # if there's any commas in text, wrap the whole text with double quotes
    if ',' in text:
        text = f'"{text}"'
    with open('votes.csv', 'a') as f:
        f.write(f'{text},{model_prediction},{gpt3_prediction},{vote}\n')


    return jsonify({'message': 'Vote successfully cast.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


##Predict

def predict_class(text, model, tokenizer):
    # Tokenize the text
    inputs = tokenizer.encode(text, return_tensors='tf')
    
    # Get model's prediction
    predictions = model(inputs)[0]
    
    # Get the predicted class index
    predicted_class_idx = tf.argmax(predictions, axis=1).numpy()[0]

    # Map the index to the class name (assuming you have a dictionary for this)
    predicted_class = label_mapping_inverse[predicted_class_idx]  # Need to define label_mapping_inverse 

    return predicted_class

label_mapping_inverse = {v: k for k, v in label_mapping.items()}
