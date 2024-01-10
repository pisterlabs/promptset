import openai
import requests
import datetime
from ai_art_creation.api.api_key import api_key
import pandas as pd

def generate_images_from_csv_file_path(csv_file_path):

    # Set your OpenAI API key
    openai.api_key = api_key
    
    prompts_df = pd.read_csv(csv_file_path)
    
    # Create an empty list to store the image paths
    image_paths = []

    for index, row in prompts_df.iterrows():

        # Set the prompt
        PROMPT = row["prompt"]

        try:

            # Set prompt and output directory
            PROMPT = PROMPT + ', in balanced composition with an abstract, geometic, original, unique, clean, clear, fractal design suitable for print on product packaging, and a color scheme that is suitable for a beer can'
            print(PROMPT)
            OUTPUT_DIR = "C:\\Users\\trent\\OneDrive\\Desktop\\Peninsula\\Can Concepts"
            #OUTPUT_DIR.mkdir(exist_ok=True)

            # Call the DALL-E API
            response = openai.Image.create(
                prompt=PROMPT,
                n=3,
                size="1024x1024",
            )

            # Get the URL of the generated image
            image_url = response["data"][0]["url"]

            # Download the image
            response = requests.get(image_url)
            
            # Get the current date and time as a timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            style = row["style"]

            # Save the image locally
            image_file = f'{OUTPUT_DIR}/{style}-{timestamp}.png'
            with open(image_file, "wb") as f:
                f.write(response.content)
            
            # Add the image path to the list
            image_paths.append(image_file)

            print(f"Image saved to: {image_file}")
            
        except Exception as e:
            print(e)
            continue
        
    # Return the list of image paths
    return image_paths

# Example usage:
#file_path = "C:\\Users\\trent\\OneDrive\\Documents\\GitHub\\ai_art_creation\\ai_art_creation\\api\\beer_can_designs.csv"
#processed_df = generate_images_from_csv_file_path(file_path)

def generate_images_from_prompt(tries = 3, PROMPT = "an illustration of a friendly bigfoot sasquatch with pine trees and mountains in a cartoon style, 8k resolution, ultra realistic, with a colorful palette"):

    # Set your OpenAI API key
    openai.api_key = api_key

    # Create an empty list to store the image paths
    image_paths = []

    for i in range(tries):
        print(i)

        try:

            # Set prompt and output directory
            PROMPT = PROMPT + ', in balanced composition with an original, unique, clean, clear, design with a pure white #000000 background.'
            print(PROMPT)
            OUTPUT_DIR = "C:\\Users\\trent\\OneDrive\\Documents\\GitHub\\trentleslie\\images\\dalle_output"
            #OUTPUT_DIR.mkdir(exist_ok=True)

            # Call the DALL-E API
            response = openai.Image.create(
                prompt=PROMPT,
                n=1,
                size="1024x1024",
            )

            # Get the URL of the generated image
            image_url = response["data"][0]["url"]

            # Download the image
            response = requests.get(image_url)
            
            # Get the current date and time as a timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Save the image locally
            image_file = f'{OUTPUT_DIR}/{timestamp}.png'
            with open(image_file, "wb") as f:
                f.write(response.content)
            
            # Add the image path to the list
            image_paths.append(image_file)

            print(f"Image saved to: {image_file}")
            
        except Exception as e:
            print(e)
            continue
            
        # Return the list of image paths
    return image_paths
    

generate_images_from_prompt()