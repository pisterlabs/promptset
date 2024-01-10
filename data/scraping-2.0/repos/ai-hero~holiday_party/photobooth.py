import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
import codenamize
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


def create_png_mask(image):
    """
    Create a PNG mask for greenscreen areas in an image.
    Args:
    image (numpy.ndarray): Input image in BGR format.

    Returns:
    numpy.ndarray: RGBA image where greenscreen areas are transparent.
    """
    # Convert BGR to HSV format
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([36, 0, 0])
    upper_green = np.array([86, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert mask: greenscreen areas become 0 (transparent)
    inverted_mask = cv2.bitwise_not(mask)

    # Convert the original image to RGBA
    rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Set the alpha channel of the image to the inverted mask
    rgba_image[:, :, 3] = inverted_mask

    return rgba_image


def replace_greenscreen_with_background(foreground, background):
    """
    Replace greenscreen areas in an image with a background image.

    Args:
    foreground (numpy.ndarray): Input image in BGR format with greenscreen.
    background (numpy.ndarray): Background image in BGR format.

    Returns:
    numpy.ndarray: Image where greenscreen areas are replaced with the background.
    """

    # Convert BGR to HSV format for the foreground image
    hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([36, 0, 0])
    upper_green = np.array([86, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert the mask: non-greenscreen areas become 0
    inverted_mask = cv2.bitwise_not(mask)

    # Apply the inverted mask to the foreground image
    foreground_masked = cv2.bitwise_and(foreground, foreground, mask=inverted_mask)

    # Apply the original mask to the background image
    background_masked = cv2.bitwise_and(background, background, mask=mask)

    # Combine the masked foreground and background
    combined = cv2.add(foreground_masked, background_masked)

    return combined


def call_dalle_api(image, mask, prompt):
    """
    Call the DALL-E API to edit an image based on the provided mask and prompt.
    Args:
    image (numpy.ndarray): Input image.
    mask (numpy.ndarray): Mask for the image.
    prompt (str): Prompt for DALL-E.

    Returns:
    str: URL of the generated image.
    """
    response = client.images.edit(
        model="dall-e-2", image=image, mask=mask, prompt=prompt, n=1, size="512x512"
    )
    return response.data[0].url


def resize_image_with_aspect_ratio(
    image, target_size=(1024, 1024), background_color=(0, 0, 0)
):
    """
    Resize an image to a target size while preserving the aspect ratio.
    Fill the extra space with a specified background color.

    Args:
    image (numpy.ndarray): The original image.
    target_size (tuple): The target size as (width, height).
    background_color (tuple): Background color in BGR format.

    Returns:
    numpy.ndarray: The resized image with preserved aspect ratio.
    """
    # Calculate aspect ratio
    h, w = image.shape[:2]
    aspect_ratio = w / h

    # Determine new dimensions based on aspect ratio
    if w > h:
        new_w = target_size[0]
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = target_size[1]
        new_w = int(new_h * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h))

    # Create a black background
    background = np.full(
        (target_size[1], target_size[0], 3), background_color, dtype=np.uint8
    )

    # Calculate center offset
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2

    # Place the resized image onto the center of the background
    background[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_image

    return background


# Streamlit UI
st.title("Greenscreen Photo Booth")

uploaded_file = st.camera_input("Take a picture")

prompt = st.text_input("Enter a prompt for the full new image:")

if uploaded_file is not None and prompt:
    # Read the uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    opencv_image_resized = resize_image_with_aspect_ratio(opencv_image)
    image_bytes = cv2.imencode(".png", opencv_image_resized)[1].tobytes()

    # Call the DALL-E API
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url

        # Display the image
        response = requests.get(image_url)
        background_img = Image.open(BytesIO(response.content))
        # Convert to cv2 image
        background_img = cv2.cvtColor(np.array(background_img), cv2.COLOR_RGB2BGR)
        processed_image = replace_greenscreen_with_background(
            opencv_image_resized, background_img
        )
        # convert cv2 image to PIL image
        img = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        st.image(img, caption="Generated Image")

        _, buffer = cv2.imencode(".png", processed_image)
        processed_image_bytes = buffer.tobytes()

        # Provide a download link for the image
        name_str = codenamize.codenamize(f"{datetime.utcnow()}")
        st.download_button(
            label="Download Image",
            data=processed_image_bytes,  # Convert your final image to bytes
            file_name=f"{name_str}.png",
            mime="image/png",
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
