import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import tensorflow_hub as tfhub
import pandas as pd
import requests
#import openai_secret_manager
from dotenv import load_dotenv
import openai
import time
import os

email_address = "octopuspaul110@gmail.com" 
st.set_page_config(page_title='Dog Vision App')
#Authentication of openai api
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
#load the model
@st.cache_resource
def load_model(model_path):
  print(f'Loading model saved to {model_path}')
  model = tf.keras.models.load_model(model_path,custom_objects = {'KerasLayer':tfhub.KerasLayer})
  print(f'Successfully loaded model saved to {model_path}')
  return model

@st.cache_data
def get_unique_breeds(path):
    label_csv = pd.read_csv(path)
    labels = np.array(label_csv['breed'])
    unique_labels = np.unique(labels)
    return unique_labels


model_path = 'saved_model/20230330-12441680180287full-image-set-imagenetv3-Adam-lr-0.0002final.h5'
model = load_model(model_path)
labels_path = 'labels/labels.csv'
dog_breeds = get_unique_breeds(labels_path)
#flask_url = ngrok.connect(5000).public_url

st.header('Welcome to the Dogify app')
st.info("""
###### *App Usage Instruction*
step1 : Take a picture of your dog and select the picture from your folder by clicking the Browse files\n
step2: click the GET BREED button to get your dog breed\n
step3: click the link below the prediction to learn more about the dog breed if you see one\n
###### *Note:*
The model cannot detect the absence of a dog in an image,Upload dog images only.
""")
#sidebar
def get_time():
  mytime = time.localtime()
  if mytime.tm_hour in [0,1,2,3,4,5,6,7,8,9,10,11]:
      hour = 'morning 👀'
  elif mytime.tm_hour in [12,13,14,15,16]:
      hour = 'afternoon 👀'
  elif mytime.tm_hour in [16,17,18,19,20,21,22,23,24]:
      hour = 'evening 👀'
  else: return None
  return hour
with st.sidebar.title('About App') as title:
  st.write(f'Hello there,Good {get_time()},this is a computer vision project that helps you get the breed and learn a few things about your dog,this project adopts the transfer learning approach in machine learning using the inceptionv3 model developed at google with 24 million Parameters,6 Billion FLOPs and trained on imagenet(more on imagenet on https://paperswithcode.com/dataset/imagenet ).\nyou can learn about the model by reading its paper on https://www.bing.com/ck/a?!&&p=883ef823742cb051JmltdHM9MTY4MDczOTIwMCZpZ3VpZD0xNzE3N2Y5NC1iOTk1LTZhNGMtMDgwNy02ZWFmYjhkODZiNDMmaW5zaWQ9NTM3Mg&ptn=3&hsh=3&fclid=17177f94-b995-6a4c-0807-6eafb8d86b43&psq=inceptionv3+paper&u=a1aHR0cHM6Ly9hcnhpdi5vcmcvcGRmLzE1MTIuMDA1NjcucGRm&ntb=1. Contact me at {email_address},Thanks...')

@st.cache_data
def predict(image_file):
    if image_file is not None:
        image = Image.open(image_file)
        image = image.resize((224,224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array,axis = 0)
        try:
          prediction = model.predict(img_array)[0]
          pred_dog_breed = dog_breeds[np.argmax(prediction)]
          confidence_score = round(np.max(prediction) * 100)
          #scorer = f"I'm {confidence_score}% sure your dog is a {pred_dog_breed}"
          return pred_dog_breed,confidence_score
        except Exception as e:
           st.write('oops 😐,you uploaded a bad image,dont upload screenshots or a blurry image,upload a clear picture of your dog.')
           os._exit()
        
def get_pred_score(pred_dog_breed,confidence_score):
    if confidence_score > 40:
      gen_pred_button = st.button("GET BREED")
      if gen_pred_button:
          st.title(f'Predicted breed for this dog is :blue[{pred_dog_breed}] :dog:')
          st.title(f'Confidence Score: {confidence_score}%')
          #st.write(f'click the botton below to learn more about your dog')
          try:
            openai_learnmore_module(pred_dog_breed)
          except Exception as e:
            if pred_dog_breed == 'pembroke':
              pred_dog_breed = pred_dog_breed + '_welsh_corgi'
            elif 'poodle' in pred_dog_breed:
              pred_dog_breed = 'poodle'
            elif 'chow' in pred_dog_breed:
              pred_dog_breed = pred_dog_breed + '_' + pred_dog_breed
            elif pred_dog_breed == 'japanese_spaniel':
              pred_dog_breed = 'japanese-chin-history-japans-royal-spaniel'
            pred_dog_breed = pred_dog_breed.replace('_','-')
            response = requests.get(f'https://www.akc.org/dog-breeds/{pred_dog_breed}')
            if response.status_code == requests.codes.ok:
              st.write(f'click the following link to learn more about _{pred_dog_breed}_ [link](https://www.akc.org/dog-breeds/{pred_dog_breed})')
            else:
              st.write(f'sorry,i cannot get any information about this dog breed right now,click this link to visit google [link](https://www.google.com/search?q={pred_dog_breed})')
    else:
            st.write(f'oh,seems like you uploaded the wrong or bad image,reload the app and upload a clear dog image.')

def openai_learnmore_module(pred_dog_breed):
    prompt = f"Tell me everything you know about {pred_dog_breed} dog breed in 20 words"
    #messages = [{"role": "system", "content" : content}]
    completion = openai.Completion.create(
        engine = "text-davinci-002",
        prompt=prompt,
        max_tokens=60,
        n =1,
        stop = None,
        temperature = 0.5
    )
    chat_Response = completion.choices[0].message.content
    if chat_Response is not None:
        st.info("{chat_Response}")
    else:
        st.write("Sorry i can't tell you more,try again later")

def mainer(image_file):
    st.image(image_file,caption='Your dog image',channels = 'RGB')
    #call the predict and explain function to make a prediction and talk about the breed
    try:
      pred_dog_breed,confidence_score = predict(image_file)
      get_pred_score(pred_dog_breed,confidence_score)
    except Exception as e:
      st.write(f'Please reload the app to use it again')

if __name__ == '__main__':
  st.write('Select an image of the dog to predict its breed')
  image_file = st.file_uploader("Image",type = ['jpg','jpeg','png'])
  if image_file is not None:
    mainer(image_file)
