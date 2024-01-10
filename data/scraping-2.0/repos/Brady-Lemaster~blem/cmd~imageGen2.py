import openai
def x(prompt, profile):
  openai.api_key = profile[2]
  response = openai.Image.create(prompt=prompt, n=profile[1], size=profile[0])
  image = response['data'][0]['url']
  return image
