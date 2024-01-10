import openai

prompt = "A women in technology conference showing a person presenting in front of a large audience."

response = openai.Image.create(prompt = prompt, n=1, size="512x512")

image = response["data"][0]["url"]

print(image)


