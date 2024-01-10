import streamlit as st
from openai import OpenAI
from util_toast import get_toast_message

def generate_image_with_dalle(description):
    """
    Generates an image using DALL-E based on a description.
    :param description: Description of the character
    :return: URL or path to the generated image
    """
    tst_start = get_toast_message("start", "Character Image")

    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
    model = "dall-e-3"
    size = "1024x1024"
    quality = "standard"

    try:
        image_response = client.images.generate(
            model=model,
            prompt=description,  # Use the function parameter
            size=size,
            quality=quality,
            n=1
        )

        tst_end = get_toast_message("end", "Character Image")
        
        # Check if the response contains the image URL
        if image_response and image_response.data:
            return image_response.data[0].url
        else:
            return None  # Or handle this case as needed
    except Exception as e:
        st.error(f"An error occurred while generating the image: {e}")
        return None
