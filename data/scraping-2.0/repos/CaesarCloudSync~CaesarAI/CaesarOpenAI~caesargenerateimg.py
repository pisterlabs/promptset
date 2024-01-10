import openai

openai.api_key = "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"
response = openai.Image.create(
  prompt="a back to the future flying car",
  n=1,
  size="1024x1024"
)

image_url = response['data'][0]['url']
print(image_url)