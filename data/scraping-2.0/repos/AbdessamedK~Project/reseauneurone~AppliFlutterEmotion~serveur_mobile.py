import os
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageOps
import numpy as np
import cv2
import torch 
from transformers import ViTForImageClassification, ViTImageProcessor
import openai
import io
import base64
openai.api_key = "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"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = ViTForImageClassification.from_pretrained("../Reconnaissance_Emotions/lebin")
processor = ViTImageProcessor.from_pretrained("../Reconnaissance_Emotions/lebin")

app = Flask(__name__)

@app.route('/process-image', methods=['POST'])


def process_image():
    print("connexion au serveur reussie")
    # Récupération de l'image depuis la requête (si nécessaire)

    image = request.files['image']
    print(type(image))

    #img = Image.open(image)
    
    # Afficher l'image avec Matplotlib
    #plt.imshow(img)
    #plt.imshow(image)
    # Vérifier si le fichier est vide
    if image.filename == '':
        return 'Fichier image vide', 400

    # Vérifier si l'extension de fichier est valide (par exemple, .jpg, .png, etc.)
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    if not image.filename.lower().endswith(tuple(allowed_extensions)):
        return 'Extension de fichier non supportée', 400
    
    img_base64 = base64.b64encode(image.read()).decode('utf-8')
    
    image_bytes = base64.b64decode(img_base64)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


     
    #mot="Aucun visage detecté"

    if len(faces) > 0:
        x, y, w, h = faces[0]
        image = image[y:y+h+30,(x-5):x+w+10]

        image = processor(images=image, return_tensors="pt")
        outputs = model(**image)
        predicted_num = torch.argmax(outputs.logits, dim=1).item()

        
        if predicted_num == 0:
            mot = "Sad"
            prompt_text = "Je suis triste, raconte moi une petite histoire pour me remonter le moral."     
        elif predicted_num == 1:
            mot = "Disgust"
            prompt_text = "Je suis dégouté, raconte moi une petite histoire pour me remonter le moral."
        elif predicted_num == 2:
            mot = "Angry"
            prompt_text = "Je suis énerver, raconte moi une petite histoire pour me calmer."
        elif predicted_num == 3:
            mot = "Neutral"
            prompt_text = "Je suis neutre, raconte moi une petite histoire pour me faire rire."
        elif predicted_num == 4:
            mot = "Fear"
            prompt_text = "J'ai peur, raconte moi une petite histoire pour me rassurer."
        elif predicted_num == 5:
            mot = "Surprise"
            prompt_text = "Je suis surpris, raconte moi une petite histoire pour me calmer."
        elif predicted_num == 6:
            mot = "Happy"
            prompt_text = "Je suis content, raconte moi une petite histoire pour me faire rire."
        else :
            mot = "pafwaaaax"

        print("Je sais ! , c'est ", mot, " !")
    
        prompt_text = "En 10 lignes maximum, " + prompt_text
        reponse = openai.chat.completions.create(
            model = "gpt-4",
            messages = [{"role" : "user", "content" : prompt_text}]
            )
        texte2 = reponse.choices[0].message.content
        reponse_chatgpt = reponse.choices[0].message.content
        print(texte2)

# Supprimer les guillemets inutiles
        texte23 = reponse_chatgpt.replace('"', '')
        print(texte23)
    # Renvoyer les deux textes à l'application Flutter
    response = {
        "texte1": mot,
        "texte2": texte2
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,threaded=False, port=5000)
