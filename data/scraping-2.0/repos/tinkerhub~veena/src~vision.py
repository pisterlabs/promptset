from openai import OpenAI
import cv2
import base64
from PIL import Image
import io


def convert_frame_to_image_data(frame):
    """
    This function converts a frame captured from a webcam to a format suitable for the OpenAI API.
    :param frame: The frame to convert.
    :return: The converted frame.
    """
    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PIL Image
    pil_image = Image.fromarray(frame)

    # Convert the PIL Image to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('ascii')


def has_speaker_left_stage(frame, client: OpenAI):
    """
    This function uses OpenAI's GPT-4 Vision model to analyze a 
    video frame and determine if a speaker has left the stage.
    
    :param frame: The video frame to analyze.
    :param client: The OpenAI client.
    :return: True if the speaker has left the stage, False otherwise.
    """
    # Convert the frame to a format suitable for the OpenAI API
    # This is a placeholder and needs to be replaced with actual code to convert the frame to a suitable format
    image_data = convert_frame_to_image_data(frame)

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "There a person in the image. True or False?"},
                    {"type": "image", "image": {"base64": image_data}},
                ],
            }
        ],
        max_tokens=300,
    )

    # Interpret the response from the OpenAI API
    is_person_left = False if response.choices[0].text == "True" else True

    # This is a placeholder and needs to be replaced with actual code to interpret the response
    return is_person_left
