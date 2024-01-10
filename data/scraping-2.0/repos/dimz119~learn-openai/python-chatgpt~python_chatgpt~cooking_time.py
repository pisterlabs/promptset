import openai
import os
import random

openai.api_key = os.getenv("OPENAI_API_KEY")

ingredient_list = [
  "kimchi",
  "ham",
  "seedweed",
  "tuna",
  "beef",
  "bacon",
  "apple",
  "onion",
  "salmon",
  "ramen"
]

def get_ingredient():
  ingredient_lst = []
  for _ in range(0, 3):
    ingredient_lst.append(
      ingredient_list[random.randint(0, len(ingredient_list)-1)])
  return ingredient_lst

lst = get_ingredient()
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=f"Given the following ingredients: {','.join(lst)} what can I cook? provide the instruction steps. Plus, provide the name of cooking title on the first line",
  max_tokens=300,
  temperature=0.7
)

print(f"Given ingrdients: {','.join(lst)}")
output = response['choices'][0]['text'].strip('\n')
output_list = output.split("\n")
title = output_list[0]
instructions = '\n'.join(output_list[1:])
print(f"Title: {title}")
print(f"{instructions}")

# Generate an image
response = openai.Image.create(
    prompt=title, # text prompt used to generate the image
    model="image-alpha-001", # DALL-E model to use for image generation.
    size="256x256",
    response_format="url"
)

# Print the URL of the generated image
print(response["data"][0]["url"])