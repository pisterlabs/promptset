import datetime
import glob
import random
import requests
import openai
import numpy as np
import openai
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image as image_
from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions


DOG_API_URL = r"https://dog.ceo/api/breeds/image/random"

model = VGG16(weights='imagenet')
MAX_TOKENS = 500


def is_working_hours():
    date = datetime.datetime.now()
    if date.weekday() in [5, 6]:
        return False
    if date.hour in list(range(6, 16)):
        return True
    return False


def get_random_jack():
    jacks_paths = glob.glob(r'C:\Users\nicks\OneDrive\Documents\Projects\SummertimeBot\static/*')
    random_jack = random.choice(jacks_paths)
    return random_jack


def get_random_dog():
    '''
    GET request to Dog API. Return random dog
    '''
    return requests.get(DOG_API_URL)


def generate_image_from_prompt(message):
    '''
    Generate image from discord message. Returns URL of image
    '''
    response = openai.Image.create(
    prompt = message.content.split("!testme")[1],
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return requests.get(image_url)


def generate_variation(n=1):
    '''
    Generate image variation from user uploaded image
    '''
    edited_image = openai.Image.create_variation(
        image=open("sample_image.png", "rb"),
        n=1,
        size="1024x1024"
        )
    return edited_image['data'][0]['url']


def classify_image(image_name='sample_image.png'):
    '''
    Classify image. Returns Tuple of (confidence score, classification)
    '''
    img = image_.load_img(image_name, color_mode='rgb', target_size=(224, 224))
    # Converts a PIL Image to 3D Numy Array
    x = image_.img_to_array(img)
    x.shape
    # Adding the fouth dimension, for number of images
    x = np.expand_dims(x, axis=0)
    #mean centering with respect to Image
    x = preprocess_input(x)
    features = model.predict(x)
    p = decode_predictions(features)
    predictions = p[0][0]
    return round(predictions[2] * 100, 2), predictions[1]


def ask_gpt(message, model_name='text-davinci-003'):
    return openai.Completion.create(
    model=model_name,
    prompt=message.content.split("!askme")[1],
    max_tokens=MAX_TOKENS,
    temperature=1
    )


def ask_turbo(message):
    """
    New turbo model added by OpenAI. Chat model, not good for creative tasks. 
    """
    return openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": f'{message.content.split("!turboask")[1]}'}
        ])
    

# TODO set flags as variables with default values so they can be changed
def save_temp_file(response):
    '''
    Saves API response to disk as 'sample_image.png'
    '''
    file = open("sample_image.png", "wb")
    file.write(response.content)
    file.close()

