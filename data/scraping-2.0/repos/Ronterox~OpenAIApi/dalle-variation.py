import os
import openai
from dallefun import download_image_from_response
from PIL import Image
import io

openai.api_key = os.getenv("OPENAI_API_KEY")

# Requeriments: Got to be a png, and X by X

name = "nickobaby"
px = 1024

# Load the image
with open(f"images/{name}.png", "rb") as image_file:
    image_data = image_file.read()
    image = Image.open(io.BytesIO(image_data))

# Convert the image to RGB format
image = image.convert("RGB")

# Convert the image to bytes
image_bytes = io.BytesIO()
image.save(image_bytes, format='PNG')
image_bytes = image_bytes.getvalue()

# Create a variation of the image
response = openai.Image.create_variation(
    image=image_bytes,
    n=3,
    size=f"{px}x{px}"
)

download_image_from_response(response, name, px)