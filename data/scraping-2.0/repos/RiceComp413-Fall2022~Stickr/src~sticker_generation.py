import requests
import replicate
import openai

def generate_dalle_sticker(query):
  response = openai.Image.create(
    prompt=query,
    n=4,
    size="512x512"
  )
  # image_url = response['data'][0]['url']
  image_urls = [response['data'][i]['url'] for i in range(4)]
  return image_urls


def generate_dalle_variations(image):
  response = openai.Image.create_variation(
    image=image,
    n=1,
    size="512x512"
  )
  image_url = response['data'][0]['url']
  return image_url

def generate_deepai_sticker(query):
  r = requests.post(
    'https://api.deepai.org/api/text2img',
    data={
      'text': query,
    },
    headers={'api-key': '8bc69e15-14aa-49ce-9b62-7a28f2efa916'}
  )
  return r.json()['output_url']

def generate_stable_diffusion_sticker(query):
  model = replicate.models.get("stability-ai/stable-diffusion")
  image_urls = model.predict(prompt=query, num_outputs=4, width = 512, height = 512)
  return image_urls
  
def generate_dummy_sticker(query):
  image_urls = [
  "https://replicate.delivery/pbxt/7TqfZvHjy7X6GiWsG7gl80UXqmbubsYO6YhGiaOPOeLDWK7PA/out-0.png",
  "https://replicate.com/api/models/stability-ai/stable-diffusion/files/19e3a073-de22-49fd-97e5-cd3942b41c9b/out-0.png",
  "https://replicate.delivery/pbxt/SzneDGsKAvRuOSKKLLvfO9uJZjD7c9PPPgBZ1RTCAhGEWK7PA/out-1.png",
  "https://replicate.delivery/pbxt/L57IAErahd40KllFsp9bsMrI2TDG5xeQnDOrt9IoMDQCLl9HA/out-3.png"
  ]
  return image_urls

def engineer_prompt(query, cutout, style):
  # Check for empty queries
  stripped_query = query.strip()
  if not stripped_query:
    return ""

  final_query = stripped_query

  # Prepend style guide
  if "sticker" not in stripped_query.lower():
    final_query = "Sticker illustration of " + final_query

  if style == 'Mono':
    final_query += ', monochrome line'
  elif style == 'Vivid':
    final_query += ', hd dramatic illustration'
  elif style == 'Abstract':
    final_query = 'Abstract ' + final_query + ' detailed sticker, artstation hd'


  # Append cutout specification
  if cutout == "circle":
    final_query += ", circle cutout"
  elif cutout == "square":
    final_query += ", square cutout"



  return final_query
