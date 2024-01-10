import streamlit as st
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import fastai.vision.all 
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from openai import OpenAI
import os
import modal 

#api boot
client = OpenAI(
    api_key=os.environ.get(st.secrets["api_key"])
)

def vit_to_string(list_obj):
  string_result = ''.join(map(str,result))
  string_result = string_result[19:]
  return string_result

#precheck function see if dog is real or not
def precheck(streeng):
  if streeng.find('dog') != -1 or streeng.find('dogs') != -1 or streeng.find('puppy') != -1 or streeng.find('puppies') != -1:
    return True
  else:
    return False

# Load your trained model
#path = path('dogdata')
#learn = load_learner(path / 'export.pkl')
# learn = tf.keras.models.load_model('./dogdata')
# learn2 = tf.keras.models.load_model('./dog-emotions-prediction')

st.title("What Up Dog")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg','png','raw','dng','jpeg'])

if uploaded_file is not None:
    # Preprocess the image
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((384, 384))  # adjust size as needed

    # Convert the image to a numpy array
    img_array = np.array(img)

    # If using TensorFlow, you might need to expand dimensions 
    # and preprocess the input depending on your model requirements
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_to_txt = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    result = img_to_txt(img)
    
    #Check if image is actually a dog
    if precheck(vit_to_string(result)) is False: 
        st.text("THAT'S NOT A DOG! (or our model didn't detect a dog, in which case we profusely apologize :( ))")
        
    #Setting dummy emotion
    default_emo = "happy"
    final_string = "You are a dog. If I were to take a picture of you right now you would be {}. Your tone and emotion would be considered {}".format(vit_to_string(result),default_emo)
    
    # GPT Prompt Init
    gpt_dog = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": final_string},
            {"role": "user", "content": "Why do you think the dog in the picture is experiencing the emotion we have labeled it with and sent to you?"},
            {"role": "user", "content": "In 1 sentence as if you were a dog: explain why you would be feeling those emotions."}] )
    
    # I beat my head against a wall to parse out the message content, but I used gpt4 to get me to this
    # Check if 'choices' exists in the response and it has at least one element
    if hasattr(gpt_dog, 'choices') and len(gpt_dog.choices) > 0:
        # Extract the 'message' dictionary from the first element of 'choices'
        # This step depends on whether 'choices' is a list of dictionaries or a list of objects
        first_choice = gpt_dog.choices[0]
        message_dict = first_choice.get('message', {}) if isinstance(first_choice, dict) else getattr(first_choice, 'message', {})

    # Extract the 'content' from the 'message' dictionary
    response_content = message_dict.get('content', '') if isinstance(message_dict, dict) else getattr(message_dict, 'content', '')

    if response_content:
        print(response_content)
    else:
        print("No content found in the response.")
    # Done inside Streamlit button function to display text only when button is pressed:
    if st.button('Make me talk'):
        st.write(response_content)