from got_survival.params import *
from openai import OpenAI
from PIL import Image
from io import BytesIO
import requests
import os
import uuid
from pathlib import Path
import pandas as pd

def create_image(
        new_character:pd.DataFrame,
        age:int,
        story:str=None,
        api_key:str=''
    ) -> Image:
    '''
    Given information about a character will create an image using the OpenAI api.
    '''
    # Instantiate OpenAI with the key
    # if api_key=='':
    #     client = OpenAI(api_key=OPENAI_API_KEY)
    # else:
    #     client = OpenAI(api_key=api_key)
    client = OpenAI(api_key=api_key)

    # Define prompt
    if story is None:
        sentence = f"""
            Character Overview:

            House: {new_character['origin'][0]}
            Age: {age}
            Popularity Index: {new_character['popularity'][0]}
            Gender: {'male' if new_character['male'][0] else 'female'}
            Nobility Status: {'Noble' if new_character['isNoble'][0] else 'Commoner'}
            Marital Status: {'married' if new_character['isMarried'][0] else 'unmarried'}

            Description:

            Create a portrait of a Game of Thrones character from
            {new_character['origin'][0]}.
            The character's age, popularity, gender, nobility status, and marital
            status should all be reflected in the image. The portrait should be
            evocative of the rich and complex world of Westeros.

            Instructions:

                Do not include any text in the image.
                Ensure the character's attire, hairstyle, and overall appearance
                align with their house, age, nobility status, and marital status.
                Capture the character's personality and the implications of their
                unique life experiences.
                Create an image that seamlessly blends into the visual aesthetic
                of Game of Thrones.,
                    quality="standard",
                    size="1024x1024",
                    n=1
            """
    else:
        sentence = f"""
            Create an image for the following story. Do not include any text or words!

            {story}
        """
    # Generate image
    response = client.images.generate(
        model="dall-e-3",
        prompt=sentence
    )

    # Extract image url
    image_url = response.data[0].url

    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Save image in processed_data/images
    # Generate a unique filename
    unique_filename = str(uuid.uuid4()) + ".png"
    folder_path = "./processed_data/images/"

    # Create the image folder if it doesn't exist
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    filename = os.path.join(folder_path, unique_filename)
    img.save(filename)

    return img , filename
