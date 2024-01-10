#uses Dalle
# import torch  # Import PyTorch
# from dalle_pytorch import OpenAIDiscreteVAE, DALLE  # Import DALLE and VAE from DALLE-pytorch
# from PIL import Image  # Import Image from PIL (Python Imaging Library)
#
# # Load the pre-trained Discrete VAE model
# vae = OpenAIDiscreteVAE()
#
# # Load the pre-trained DALL-E model
# dalle_path = 'path/to/dalle.pt'
# dalle = DALLE(vae=vae).load(dalle_path)
#
# def generate_images(text, num_images=1):
#     """
#     Generate images from the given text using the pre-trained DALL-E model.
#
#     :param text: The input text to generate images from
#     :param num_images: The number of images to generate (default: 1)
#     :return: A list of PIL Image objects
#     """
#     # Tokenize the input text
#     tokens = dalle.tokenizer.tokenize([text], 128).squeeze(0)
#
#     # Generate images using the DALL-E model
#     images = dalle.generate_images(tokens, num_images=num_images)
#
#     # Convert the generated images to PIL Image objects
#     images = [Image.fromarray(image.permute(1, 2, 0).clamp(0, 1).numpy()) for image in images]
#
#     return images

# uses imagen model
import torch
from imagen_pytorch import Imagen
from PIL import Image

# Load the pre-trained Imagen model
imagen = Imagen()

def generate_images(text, num_images=1):
    """
    Generate images from the given text using the pre-trained Imagen model.

    :param text: The input text to generate images from
    :param num_images: The number of images to generate (default: 1)
    :return: A list of PIL Image objects
    """
    # Tokenize the input text
    tokens = imagen.tokenizer.tokenize([text], 128).squeeze(0)

    # Generate images using the Imagen model
    images = imagen.generate_images(tokens, num_images=num_images)

    # Convert the generated images to PIL Image objects
    images = [Image.fromarray(image.permute(1, 2, 0).clamp(0, 1).numpy()) for image in images]

    return images