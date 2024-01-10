import openai
import requests
import os

# Replace with your OpenAI API key
api_key = "YOUR_OPENAI_API_KEY"

# Initialize the OpenAI API client
openai.api_key = api_key

# Prompt for generating the image
prompt = input("Enter a prompt to generate an image: ")

try:
    # Generate an image
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"  # You can adjust the size as needed
    )

    # Extract the image URL from the response
    image_url = response['data'][0]['url']

    # Fetch the image from the URL
    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        # Specify the save path
        save_path = r"Specify the path you want to save the image to"

        # Save the image to the specified path
        with open(save_path, "wb") as image_file:
            image_file.write(image_response.content)
        print(f"Image generated and saved at {save_path}")
    else:
        print(f"Failed to fetch the image: {image_response.status_code}")
except openai.error.OpenAIError as e:
    print(f"OpenAI API error: {e}")
except Exception as ex:
    print(f"An error occurred: {ex}")

