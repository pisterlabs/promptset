from utils import draw_circle, encode_image
import instructor
from instructor import Mode
from openai import OpenAI
from pydantic import BaseModel, Field
from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

client = instructor.patch(OpenAI(), mode=Mode.MD_JSON)

class ObjectDetection(BaseModel):
    """
    You are an object detection expert.
    Find object in image. Top left of the image is [0, 0].
    For cases involving the identification of people or animals, 
    focus on locating and identifying the face of the person or animal.
    """
    x: int  = Field(description="x coordinate of detected object", default=0)
    y: int = Field(description="y coordinate of detected object", default=0)
    object_found_details: str = Field(description="Details of detected object.", default="")
    image_description: str = Field(descripion="Description of image.", default="")


def ask_gpt4_vision(system_instrutions, question, image_path):
    base64_image = encode_image(image_path)

    detected = client.chat.completions.create(
        response_model=ObjectDetection,
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
                    {"type": "text", "text": question},
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
    pprint(detected.model_dump_json())
    return {"x": detected.x, "y": detected.y}

# image_path = "assets/kitten-and-puppy.webp"
# image_path = "assets/puppy.jpg"
image_path = "assets/fire.png"

system_instructions = """You are an image recognition expert."""

question = "Detect Fire"
coordinates = ask_gpt4_vision(system_instructions, question, image_path)
draw_circle(image_path, coordinates)
