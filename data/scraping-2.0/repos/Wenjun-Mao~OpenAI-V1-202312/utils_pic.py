# utils_pic.py:

from PIL import Image
import os
import base64
import requests

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def create_expanded_and_mask_images(base_image_path, res):
    """
    Function to create an expanded image with a white background and a mask image.

    Parameters:
    - base_image_path: str, the path to the base image.
    - res: int, resolution of the output images, can be 256, 512, or 1024.

    Returns:
    - expanded_image_path: str, the path to the expanded image.
    - mask_image_path: str, the path to the mask image.
    """
    # Validate the resolution
    if res not in [256, 512, 1024]:
        raise ValueError("Resolution must be one of 256, 512, or 1024.")

    # Load the image
    base_image = Image.open(base_image_path)

    # Convert to PNG if not already in PNG format
    if base_image.format != "PNG":
        base_image = base_image.convert(
            "RGBA"
        )  # Convert to RGBA to ensure transparency support

    # Extract the base filename without the extension and directory of the input file
    base_filename = os.path.splitext(os.path.basename(base_image_path))[0]
    input_dir = os.path.dirname(base_image_path)

    # Create a new image with white background and the specified resolution
    new_image = Image.new("RGB", (res, res), "white")
    new_image.paste(
        base_image,
        (int((res - base_image.width) / 2), int((res - base_image.height) / 2)),
    )

    # Save the new image as PNG in the same directory as the input file
    expanded_image_path = os.path.join(input_dir, f"{base_filename}_expanded_{res}.png")
    new_image.save(expanded_image_path, "PNG")

    # Create a mask with transparent areas where the original image is not present
    mask = base_image.split()[-1].point(lambda x: 255 if x > 0 else 0)
    mask_image = Image.new("RGBA", (res, res), (0, 0, 0, 0))
    mask_image.paste(
        mask,
        (int((res - base_image.width) / 2), int((res - base_image.height) / 2)),
        mask=mask,
    )

    # Save the mask image in the same directory as the input file
    mask_image_path = os.path.join(input_dir, f"{base_filename}_mask_{res}.png")
    mask_image.save(mask_image_path, "PNG")

    return expanded_image_path, mask_image_path


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_img_desc(image_path):
    # Encode the image
    base64_image = encode_image(image_path)

    # Set the headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client.api_key}",
    }

    # Set the payload
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in the image, do not use full sentences, just describe the objects, ingore any watermarks or text on the image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 65,
    }

    # Send the request
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    # Get the image description
    img_desc = response.json()["choices"][0]["message"]["content"]

    return img_desc


def get_expanded_img_url(
    expanded_image_path, mask_image_path, img_desc, n=1, size="1024x1024"
):
    pics = client.images.edit(
        image=open(expanded_image_path, "rb"),
        mask=open(mask_image_path, "rb"),
        prompt=img_desc,
        n=n,
        size=size,
    )

    url = pics.data[0].url

    return url
