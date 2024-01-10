#!/usr/bin/env python3

import argparse
import requests
import os
from openai import OpenAI

def main(model_gen, prompt_gen, size_gen, quality_gen, n_gen, name_gen):
  client = OpenAI()

  response = client.images.generate(
    model=model_gen,
    prompt=prompt_gen,
    size=size_gen,
    quality=quality_gen,
    n=n_gen
  )

  # Retrieve the image URL
  image_url = response.data[0].url

  # Download the image
  r = requests.get(image_url)

  # Check if images/ directory exists
  if not os.path.exists("images/"):
    os.mkdir("images/")

  # Create file name from prompt
  file_name = name_gen.replace(" ", "_")
  file_name += ".jpg"

  # Create full path
  file_path = "images/" + file_name

  # Check if the image was downloaded successfully
  if r.status_code == 200:
    # Save the image to a file
    with open(file_path, "wb") as f:
      f.write(r.content)
  else:
    print("Error downloading image")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate and download images using OpenAI.")
  parser.add_argument("prompt", type=str, help="The prompt for image generation.")
  parser.add_argument("name", type=str, help="The name of the generated image.")
  parser.add_argument("--model", default="dall-e-3", type=str, help="The model to use for image generation. Default is 'dall-e-3'.")
  parser.add_argument("--size", default="1024x1024", type=str, help="The size of the generated image. Default is '1024x1024'.")
  parser.add_argument("--quality", default="standard", type=str, help="The quality of the generated image. Default is 'standard'.")
  parser.add_argument("--n", default=1, type=int, help="The number of images to generate. Default is 1.")

  args = parser.parse_args()

  try:
    main(args.model, args.prompt, args.size, args.quality, args.n, args.name)
  except Exception as e:
    print("Error:", str(e))
    parser.print_help()
