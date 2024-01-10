from utils.main_utils import *
from PIL import Image
from io import BytesIO
import openai


async def generate_variation(model: str, path_image: str) -> str:
    """
    This function generates a variation of an image based on the provided model and image path.

    Parameters:
    model (str): The model to be used for image variation. It can be "dall-e-2".
    path_image (str): The path to the image to be varied.

    Returns:
    str: The URL of the varied image. If an error occurs during image variation, it returns a string starting with "Error:" followed by the error message.
    """
    try:
        # Open the image file
        image = Image.open(path_image)
        # Resize the image to 256x256 pixels
        width, height = 256, 256
        image = image.resize((width, height))
        # Convert the image to a byte stream
        byte_stream = BytesIO()
        image.save(byte_stream, format='PNG')
        byte_array = byte_stream.getvalue()
        # Generate the image variation using the OpenAI API
        response = client.images.create_variation(
            image=byte_array,
            model=model,
            n=1,
            size="1024x1024"
        )
        # Return the URL of the varied image
        return response.data[0].url
    except openai.OpenAIError as e:
        # If an error occurs, return the error message
        return f"Error: {str(e)}"
