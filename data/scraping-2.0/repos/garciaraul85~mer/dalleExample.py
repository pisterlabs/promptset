import openai

client = openai.OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="A cyborg cowboy in a western cyberpunk environment",
    size="1024x1024",
    quality="standard",
    n=1 # number of images I want to generate
)

image_url = response.data[0].url
print(image_url)