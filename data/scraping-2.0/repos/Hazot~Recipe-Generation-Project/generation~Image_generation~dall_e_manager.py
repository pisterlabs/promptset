#generations
import openai
import os
def initializeDallE():
  key = os.environ.get('OPENAI_KEY')
  openai.api_key = key
  openai.Model.list()
def generateImage(prompt):
  response = openai.Image.create(
    prompt=prompt,
    n=1,
    size="256x256"
  )
  image_url = response['data'][0]['url']
  print(image_url)
  return response

