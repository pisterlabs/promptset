import openai
import os
os.environ["OPENAI_API_KEY"] = "sk-"
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.Image.create(
  prompt="A round badge is displayed completely in the center, with a cogwheel decoration on the periphery and a maple leaf logo in the middle part, with a popular color scheme of the 1950s",
  n=5,
  size="512x512"
)
i=0
while i < 5:
  image_url = response['data'][i]['url']
  print (image_url)
  i = i+1