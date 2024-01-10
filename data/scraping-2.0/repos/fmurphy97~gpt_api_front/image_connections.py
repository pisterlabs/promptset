import openai
import requests


def generate_images(prompt_message, num_images=4, img_size='256x256'):
    """
    Generates images based on an input text
    num_images: number of images we want to generate
    img_size: size of the output images, must be on of ['256x256', '512x512', '1024x1024']
    prompt_message: based on which text we want to generate images
    """
    return openai.Image.create(prompt=prompt_message, n=num_images, size=img_size)


def get_images(response):
    """Based on a response plots images"""
    resulting_images = []
    for i, resp_i in enumerate(response['data']):
        image_url = resp_i['url']
        resulting_images.append(image_url)

    return resulting_images


def generate_image_variations(response, resp_id, num_images=4, img_size='256x256'):
    """
    Generates images based on an input image
    num_images: number of images we want to generate
    img_size: size of the output images, must be on of ['256x256', '512x512', '1024x1024']
    prompt_message: based on which text we want to generate images
    """
    return openai.Image.create_variation(
        image=requests.get(response['data'][resp_id]["url"]).content,
        n=num_images,
        size=img_size
    )
