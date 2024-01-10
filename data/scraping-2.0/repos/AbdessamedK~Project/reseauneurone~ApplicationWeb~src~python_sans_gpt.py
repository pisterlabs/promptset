import sys
import base64
import cv2
import numpy as np
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import openai

model = ViTForImageClassification.from_pretrained("../../Reconnaissance_Emotions/lebin")
processor = ViTImageProcessor.from_pretrained("../../Reconnaissance_Emotions/lebin")

args_from_node = sys.argv[1:]
base64_image = args_from_node[0]

image_bytes = base64.b64decode(base64_image)
image_np = np.frombuffer(image_bytes, dtype=np.uint8)
image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

image = processor(images=image, return_tensors="pt")
outputs = model(**image)
predicted_num = torch.argmax(outputs.logits, dim=1).item()

mot=""

if predicted_num == 0:
    mot = "Triste"
    prompt_text = "Je suis triste, raconte moi une petite histoire pour me remonter le moral. Ta réponse aura la forme 'Tu es triste | (raconte l'histoire ici)'"     
elif predicted_num == 1:
    mot = "Degout"
    prompt_text = "Je suis dégouté, raconte moi une petite histoire pour me remonter le moral. Ta réponse aura la forme 'Tu es dégouté | (raconte l'histoire ici)'"
elif predicted_num == 2:
    mot = "Enerver"
    prompt_text = "Je suis énerver, raconte moi une petite histoire pour me calmer. Ta réponse aura la forme 'Tu es énerver | (raconte l'histoire ici)'"
elif predicted_num == 3:
    mot = "Neutre"
    prompt_text = "Je suis neutre, raconte moi une petite histoire pour me faire rire. Ta réponse aura la forme 'Tu es neutre | (raconte l'histoire ici)'"
elif predicted_num == 4:
    mot = "Peur"
    prompt_text = "J'ai peur, raconte moi une petite histoire pour me rassurer. Ta réponse aura la forme 'Tu as peur | (raconte l'histoire ici)'"
elif predicted_num == 5:
    mot = "Surpris"
    prompt_text = "Je suis surpris, raconte moi une petite histoire pour me calmer. Ta réponse aura la forme 'Tu es surpris | (raconte l'histoire ici)'"
elif predicted_num == 6:
    mot = "Content"
    prompt_text = "Je suis content, raconte moi une petite histoire pour me faire rire. Ta réponse aura la forme 'Tu es content | (raconte l'histoire ici)'"
else :
    mot = "pafwaaaax"
    
print(mot + "| Il était une fois ...")



