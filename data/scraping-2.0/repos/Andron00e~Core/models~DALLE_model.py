import openai
import numpy as np
import imageio
from config import GPT_API_KEY
from abstract_model import OmniModel
from PIL import Image
import requests
from io import BytesIO
import time

class DALLE_model(OmniModel):
  API_KEY = GPT_API_KEY

  def __init__(self):
    super().__init__()
    
    openai.api_key = self.API_KEY
    self.input_type = ['text']
    self.output_type = ['photo']
    
    self.discription = 'image model'
    self.model_label = 'dalle'
    
    self.imsize = '1024x1024'
    
  def predict(self, prompt, history=[]):
    response = openai.Image.create(
      prompt=prompt,
      n=1,
      size=self.imsize
    )
    image_url = response['data'][0]['url']
    image = self.url_to_pil(image_url)
    filename = f'C:/Users/Reny/Documents/GitHub/Core/photos/{time.time()}.png'
    image.save(filename)
    return [filename]

  def url_to_pil(self, url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

 
if __name__ == '__main__':
  model = DALLE_model()
  model.render('hello')