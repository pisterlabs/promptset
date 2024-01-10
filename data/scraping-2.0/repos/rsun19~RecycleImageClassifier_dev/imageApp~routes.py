import io
from flask import Flask, jsonify, request, render_template, send_file
from imageApp import app
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
import torchvision.models as models
import random
import json
import nltk
from nltk.stem.porter import PorterStemmer
import os
import openai
from imageApp import GptKey
import matplotlib.pyplot as plt
from ultralytics import YOLO
import base64

openai.organization = GptKey.organization_key
openai.api_key = GptKey.api_key
openai.Model.list()

stemmer = PorterStemmer()
nltk.download('punkt')
def tokenize(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
  tokenized_sentence = [stem(w) for w in tokenized_sentence]
  bag = np.zeros(len(all_words), dtype=np.float32)
  for idx, w in enumerate(all_words):
    if w in tokenized_sentence:
      bag[idx] = 1.0
  return bag

BASE = "https://127.0.0.1:5000/"

weights = torchvision.models.ResNet18_Weights.DEFAULT
auto_transforms = weights.transforms()

class AlexNet(nn.Module):
  def __init__(self):
    super(AlexNet, self).__init__()
    self.network = models.resnet18(weights=weights).eval()
    last_layer = self.network.fc.in_features
    self.network.fc = nn.Linear(last_layer, 6)
  
  def forward(self, x):
    return self.network(x)

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.l2 = nn.Linear(hidden_size, hidden_size)
    self.l3 = nn.Linear(hidden_size, num_classes)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    out = self.relu(out)
    out = self.l3(out)
    return out
  
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

model = AlexNet()
model.load_state_dict(torch.load("imageApp/cifar_net (4).pth"))
#model = torch.jit.load('imageApp/cifar_net.pth')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

yolo_model = YOLO('imageApp/last.pt')

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp']

@app.route("/api/car", methods=["PUT", "OPTIONS"], endpoint="car")
def detect_cars():
    try:
      file = request.files['image']
      filename = file.filename
      if (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.webp') or filename.endswith('.heic')):
        image_stream = file.read()
        img = Image.open(io.BytesIO(image_stream))
        results = yolo_model(img)
        for r in results:
          im_array = r.plot()
          im = Image.fromarray(im_array[..., ::-1])
          return jsonify({"images": image_to_base64(im)}), 200
      else:
        return jsonify({"message": "File needs to be jpg, jpeg, png, heic, or webp"})
    except Exception as e:
       return jsonify({"message": str(e)}), 404

@app.route("/car-results", methods=['POST'], endpoint = "carResult")
def car_results():
    # Get uploaded image file
    if 'carImage' in request.files:
      car_image_file = request.files['carImage']
      filename = car_image_file.filename
      if (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.webp') or filename.endswith('.heic')):
        img = Image.open(car_image_file)
        results = yolo_model(img)
        for r in results:
          im_array = r.plot()
          im = Image.fromarray(im_array[..., ::-1])
          data = io.BytesIO()
          im.save(data, "JPEG")
          encoded_img_data = base64.b64encode(data.getvalue())
          return render_template("carresults.html", img_data=encoded_img_data.decode('utf-8'))

@app.route("/api/chat/<params>", methods=['GET'], endpoint='chat')
def chat(params):
    if request.method == 'GET':
        with open('imageApp/prompts.json', 'r') as json_data:
            intents = json.load(json_data)
        
        data = torch.load('imageApp/chatbot.pth')

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data['all_words']
        tags = data['tags']
        model_state = data["model_state"]
        model_1 = NeuralNet(input_size, hidden_size, output_size).to(device)
        model_1.load_state_dict(model_state)
        model_1.eval()
        sentence = " ".join(params.strip().split('+'))
        ai_sentence = params.replace("+", " ")
        #return jsonify({"response": sentence})
        sentence = tokenize(str(sentence))
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model_1(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return jsonify({'response': f"{random.choice(intent['responses'])}"}), 200
        else:
            messages = [{"role": "assistant", "content": ai_sentence }]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages
            )
            response_message = response['choices'][0]['message']['content']
            return jsonify({'response': response_message, 'hi': messages}), 200

@app.route("/results", methods=['POST'])
def results():
    # Get uploaded image file
    image = request.files['image']

    # Process image and make prediction
    image_tensor = process_image(Image.open(image))
    output = model(image_tensor)

    # Get class probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)
    probabilities = probabilities.detach().numpy()[0]

    # Get the index of the highest probability
    class_index = probabilities.argmax()

    # Get the predicted class and probability
    predicted_class = waste_types[class_index]
    predicted_class = str(predicted_class).capitalize()
    probability = probabilities[class_index]

    # Sort class probabilities in descending order
    class_probs = list(zip(waste_types, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)

    # Render HTML page with prediction results
    return render_template('results.html', class_probs=class_probs,
                           predicted_class=predicted_class, probability=probability*100)

def process_image(image):
    transformation = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    image = image.resize((384, 512))
    #image = image.resize((512, 384))
    #image_tensor = transformation(image)
    image_tensor = auto_transforms(image).unsqueeze(0)
    #image_tensor = weights.transforms(image)
    #image_tensor = transformation(image).unsqueeze(0)
    return image_tensor

waste_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']