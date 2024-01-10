import openai
import requests
import os
from PIL import Image
from datetime import datetime

def text2image(text, show=False):
    # Call the OpenAI DALL-E2 API to generate the image
    response = openai.Image.create(
        prompt=text,
        size='1024x1024',
        model='image-alpha-001'
    )

    # Get the image data from the API response
    url = response['data'][0]['url']
    image_data = requests.get(url).content
    image_save_folder = "./images/out"
    now = datetime.now()
    image_path = os.path.join(image_save_folder, 'de2_' + now.strftime('%Y-%m-%d_%H-%M-%S') + '.png')
        
    if show:
        with open(image_path, 'wb') as f:
            f.write(image_data)
        # Load the image to a PIL Image object and display it
        image = Image.open(image_path)
        image.show()
    
    return image_path


if __name__ == '__main__':
    text2image("A dog eating pizza", show=True)
