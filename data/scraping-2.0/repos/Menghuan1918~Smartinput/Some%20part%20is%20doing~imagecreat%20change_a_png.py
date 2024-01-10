import openai
import os
os.environ["OPENAI_API_KEY"] = "sk-"
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.Image.create_variation(
  image=open("2.png", "rb"),
  n=3,
  size="256x256"
)
i=0
while i < 3:
  image_url = response['data'][i]['url']
  print (image_url)
  i = i+1