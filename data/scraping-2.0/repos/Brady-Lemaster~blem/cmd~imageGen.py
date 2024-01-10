import openai
def x(image, key):
  openai.api_key = key
  response = openai.Image.create(prompt=image, n=1, size="256x256")
  image = response['data'][0]['url']
  return image
