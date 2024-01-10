import openai
import os

# Asking the user for a string input
user_input = input("Please describe the image you want to generate: ")

# Displaying the user's input
print("You entered:", user_input)

openai.api_key = os.getenv("OPENAI_API_KEY")


response = openai.Image.create(
  model="dall-e-3",
  prompt=user_input,
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url

print(image_url)
