from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import  VGG19
import openai

openai.api_key = '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'





app = FastAPI()

@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
    model = tf.keras.models.load_model('./6claass.h5')
    img = cv2.imread(file)
    img = cv2.resize(img, (180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    vgg_model = VGG19(weights = 'imagenet',  include_top = False, input_shape = (180, 180, 3)) 
    for layer in vgg_model.layers :
        layer.trainable = False
    img=vgg_model.predict(img)
    img=img.reshape(1,-1)
    pred = model.predict(img)[0]
    predicted_class = np.argmax(pred)
    cat=['Tinea Ringworm Candidiasis and other Fungal Infections','Seborrheic Keratoses and other Benign Tumors','Melanoma Skin Cancer Nevi and Moles','Herpes HPV and other STDs ','Eczema','Acne and Rosacea ']
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=f"give me possible home remedies for {cat[predicted_class]}",
    max_tokens=60
  )
    return response.choices[0].text.strip()
    
    




@app.get("/predict")
async def get_prediction():
    prediction = upload_file()
    return {"prediction": prediction}
    



