from openai import OpenAI
import base64
from PIL import Image
import io
import os 
from dotenv import load_dotenv
import subprocess

load_dotenv()
client = OpenAI()

path = "/home/pk/PK2.0/photo.jpg"

def take_photo(photo_path):
    # Command to take a photo with libcamera-still
    command = ['libcamera-still', '-o', photo_path]

    # Run the command
    subprocess.run(command)

    print(f"Photo taken and saved to {photo_path}")

# Example usage
take_photo(path)


def process_image(image_path):
  # Resize the image
  peppers = Image.open(image_path)
  image_size = peppers.size
  peppers.resize((image_size[0] // 2, image_size[1] // 2)).save("mini_peppers.jpg", optimize=True, quality=95)
  
  print(f"Image resized to {peppers.size}")
  # peppers.resize((216, 216)).save("mini_peppers.jpg", optimize=True, quality=95)

  # Function to encode image to base64
  def encode_image_to_base64(image_path):
    with Image.open(image_path) as image:
      buffered = io.BytesIO()
      image.save(buffered, format="JPEG")
      return base64.b64encode(buffered.getvalue()).decode()

  # Encode the image
  encoded_image = encode_image_to_base64("mini_peppers.jpg")

  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "You are an expert in identifying items of food from images. List the food items in this image. Give responses only in the format 'Food item: [food item]'"},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{encoded_image}",
            },
          },
        ],
      }
    ],
    max_tokens=300,
  )

  return response.choices[0].message.content


print(process_image(path))