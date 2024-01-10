import os
from openai import OpenAI
from requests import get
from .base_model import Model

api_key = os.environ.get("OPENAI_API_KEY")

class DallE(Model):
  def __init__(self):
    self.client = OpenAI(api_key=api_key)
  
  def create(self, prompt, directory):
    response = self.client.images.generate(
      model="dall-e-3",
      prompt=prompt,
      size="1792x1024",
      quality="standard",
      n=1,
    )

    with open(directory + "/image.png", "wb") as file:
      response = get(response.data[0].url)
      file.write(response.content)
