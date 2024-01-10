from openai import OpenAI
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="帮我画一个圣诞树，要求符合圣诞节气氛",
  size="1792x1024", 
  quality="standard",
  n=1,
)

image_url = response.data[0].url

print(image_url)