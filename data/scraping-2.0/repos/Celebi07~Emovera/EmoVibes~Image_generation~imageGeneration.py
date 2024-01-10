import openai
openai.api_key = open("Image_generation\API_KEY", "r").read()
response = openai.Image.create(
  prompt="5 yellow dogs",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url) 