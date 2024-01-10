from datetime import datetime
from openai import OpenAI
from PIL import Image
import io, base64

output_folder = "GeneratedImages"
output_b64_json_folder = "b64_json"
output_jpeg_folder = "jpeg"

def generate_dalle_image_from_prompt(client: OpenAI, promp, size="1024x1024"):
    # size:  1024x1024 | 1024x1792 | 1792x1024 
    # Quality: standard | hd
    # n: number of images generated

    image_object = client.images.generate(
        model="dall-e-3",
        response_format="b64_json",
        prompt=promp,
        size=size,
        n=1
    )
    
    stored_file_jpeg_path = store_b64_json_as_file(
        image_object.data[0].b64_json, 
        promp
    )

    return stored_file_jpeg_path


def store_b64_json_as_file(b64_json_string, promp):
    # create unique file path
    current_datetime = datetime.now()
    date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_file_path = promp[:10].replace(" ", "_") + date_time_string

    b64_json_file_path = f"{output_folder}/{output_b64_json_folder}/{output_file_path}.txt"
    jpeg_file_path = f"{output_folder}/{output_jpeg_folder}/{output_file_path}.jpeg"

    # store b64_json in b64_json folder
    with open(b64_json_file_path, 'w') as output_file:
        output_file.write(b64_json_string)
    
    # store b64_json as jpeg image in jpeg folder
    with open(b64_json_file_path, 'r') as input_file:
        text_content = input_file.read()
        # can add b64_json_string instead of text_content if we dont need to store the b64_json file
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(text_content, "utf-8"))))
        img.save(jpeg_file_path)

    return jpeg_file_path