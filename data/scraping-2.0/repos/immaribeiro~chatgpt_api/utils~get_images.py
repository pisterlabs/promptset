import requests
import json
import os
import openai
import time

from PIL import Image
from io import BytesIO
from utils.logger import get_logger

def download_images(response, filename_url, output_folder):
    logger = get_logger(__name__, 'utils/download_images.log')
    try:
        os.makedirs(output_folder, exist_ok=True)
        images_output_path = output_folder
        image_counter = 0    
        for image in response['data']:
            url = image['url']
            response = requests.get(url)
            image_counter += 1
            current_time = str(int(time.time()))
            prompt_image_short = filename_url[:min(len(filename_url), 30)]
            filename = prompt_image_short.replace(" ", "_") + '_' + str(image_counter) + '_' + current_time +'.png'

            with open(images_output_path + filename, 'wb') as f:
                f.write(response.content)
            logger.info('New image file downloaded: ' + filename)
        
        return f
    except Exception as e:
        logger.debug(f'An error occurred while downloading images: {str(e)}')

def generate_json(response, prompt, json_output_folder):
    logger = get_logger(__name__, 'utils/generate_json.log')
    try:
        os.makedirs(json_output_folder, exist_ok=True)
        response_json = json.dumps(response, indent=4)
        current_time = str(int(time.time()))
        prompt_image_short = prompt[:min(len(prompt), 30)]
        filename = prompt_image_short.replace(" ", "_") + '_' + current_time + '.json'
        with open(json_output_folder + filename , 'wb') as f:
                f.write(response_json.encode())
        logger.info('New image request successfully created. \nSaved as json file' + filename)
        return f

    except Exception as e:
        logger.debug(f'An error occurred while generating json response file: {str(e)}')

def resize_image(image):
    # Read the image file from disk and resize it
    width, height = 256, 256
    image = image.resize((width, height))

    # Convert the image to a BytesIO object
    byte_stream = BytesIO()
    image.save(byte_stream, format='PNG')
    image_byte_array = byte_stream.getvalue()

    return image_byte_array

def get_images(prompt=None, number=None, size=None, response_format=None, download=None):
    logger = get_logger(__name__, 'utils/get_images.log')
    try:
        os.makedirs('utils/configs/images', exist_ok=True)
        image_output_folder = 'output/images/'
        json_output_folder = 'output/json/images/'

        # Load default values from json file
        with open('utils/configs/images/get_images_config.json') as f:
            config = json.load(f)
                
        args = {
            'prompt': prompt or config.get('prompt'),
            'n': number or config.get('number'),
            'size': size or config.get('size'),
            'response_format': response_format or config.get('response_format'),
        }
        args = {k: v for k, v in args.items() if v is not None}
        
        response = openai.Image.create(**args)

        download = config.get('download')
        if download is True:
            download_images(response, prompt, image_output_folder)
            generate_json(response, prompt, json_output_folder)
            logger.info(f'Images and JSON file successfully downloaded!')

        else:
            generate_json(response, prompt, json_output_folder)
            logger.info(f'JSON file successfully generated!')
        
        time.sleep(0.1)
        return response

    except openai.error.OpenAIError as e:
        logger.debug(f'An error occurred while generating images: {str(e)}')

def get_image_variations(image=None, number=None, size=None, response_format=None, download=None):
    logger = get_logger(__name__, 'utils/get_image_variations.log')
    try:
        os.makedirs('utils/configs/images', exist_ok=True)
        image_output_folder = 'output/images/variations/'
        json_output_folder = 'output/json/images/variations/'
        image_filename = os.path.basename(image)

        # Load default values from json file
        with open('utils/configs/images/get_images_config.json') as f:
            config = json.load(f)

        with Image.open(image) as image:
            image = resize_image(image)
                
        args = {
            'image' : image or config.get('image'),
            'n': number or config.get('number'),
            'size': size or config.get('size'),
            'response_format': response_format or config.get('response_format'),
        }
        args = {k: v for k, v in args.items() if v is not None}
        
        response = openai.Image.create_variation(**args)

        download = config.get('download')
        if download is True:
            download_images(response, image_filename, image_output_folder)
            generate_json(response, image_filename, json_output_folder)
            logger.info(f'Images and JSON file successfully downloaded!')

        else:
            generate_json(response, image_filename, json_output_folder)
        logger.info(f'JSON file successfully generated!')

        return response

    except openai.error.OpenAIError as e:
        logger.debug(f'An error occurred while generating  images: {str(e)}')

#def create_image_edit(prompt=None, image=None, mask=None, size=None, response_format=None, number=None, download=None):
