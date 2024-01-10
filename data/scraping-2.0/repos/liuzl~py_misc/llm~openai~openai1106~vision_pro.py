# encoding: UTF-8
import os
import openai
import json
import base64
from PIL import Image, ImageDraw

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def draw_circle(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    radius = 5
    x = coordinates['x']
    y = coordinates['y']

    draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)], 
        outline='red', 
        width=4
    )

    new_image_path = image_path.split('.')[0] + '_detected.' + image_path.split('.')[1]

    image.save(new_image_path)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

def ask_gpt4_vision(system_instrutions, object_to_detect, image_path):
    base64_image = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            max_tokens=100,
            messages=[
                {
                    "role": "system", 
                    "content": system_instrutions
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Detect: {object_to_detect}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        content = response.choices[0].message.content
        json_str = content.strip('`json\n') # Extract the JSON part from the string (remove the ```json and ``` at both ends)
        coordinates = json.loads(json_str) # Convert the JSON string into a Python dictionary

        print('-' * 50)
        print("Detect:", object_to_detect)
        print("Details:", coordinates["details"])
        print(f"Coordinates: [{coordinates['x']}, {coordinates['y']}]")
        print('-' * 50)

    except Exception as e:
        print(e)
        coordinates = {"x": 0, "y": 0, "details": ""}
    
    return coordinates

image_path = "two_dogs.jpeg"

system_instructions = """
As an image recognition expert, your task is to analyze images and provide 
output in JSON format with the following keys only: 'x', 'y', and 'details'.

- 'x' and 'y' should represent the coordinates of the center of the detected 
object within the image, with the reference point [0,0] at the top left corner.
- 'details' should provide a brief description of the object identified in the image.

For cases involving the identification of people or animals, focus on locating and 
identifying the face of the person or animal. Ensure that the given 'x' and 'y' 
coordinates correspond to the center of the identified face.

Please adhere strictly to this output structure:
{
  "x": value,
  "y": value,
  "details": "Description"
}

Note: Do not include any additional data or keys outside of what has been specified.
"""

detect = "brown dog's nose"

coordinates = ask_gpt4_vision(system_instructions, detect, image_path)
draw_circle(image_path, coordinates)