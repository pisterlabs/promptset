from utils.main_utils import *
import openai


async def generate_image(prompt: str, model: str) -> str:
    """
    This function generates an image based on the provided prompt and model.

    Parameters:
    prompt (str): The prompt to be used for image generation.
    model (str): The model to be used for image generation. It can be "dall-e-3" or "dall-e-2".

    Returns:
    str: The URL of the generated image. If an error occurs during image generation, it returns a string starting with "Error:" followed by the error message.
    """
    try:
        # Generate the image using the OpenAI API
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        # Return the URL of the generated image
        return response.data[0].url
    except openai.OpenAIError as e:
        # If an error occurs, return the error message
        return f"Error: {str(e)}"
