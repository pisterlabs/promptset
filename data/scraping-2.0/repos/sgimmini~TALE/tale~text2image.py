from diffusers import DiffusionPipeline, DDIMScheduler, KDPM2AncestralDiscreteScheduler
import torch
from .pre_processor import PreProcessor
from openai import OpenAI
import base64
import requests
from PIL import Image
from tqdm import tqdm
import os

class Text2Image:
    """
    This class takes a prompt as input and returns an image as output.
    """

    def __init__(self) -> None:
        device = None
        self.pipe = None
        self.client = None
        client = None

        is_openai = False
        is_diffusion = False

    def init_diffusion(self):
        """
        This function initializes the diffusion pipeline.
        :return: None
        """
        device = "cuda"
        self.pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(device)
        self.pipe.safety_checker = None
        self.pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )
        return

    def init_openai(self):
        """
        This function initializes the openai api.
        :return: None
        """
        # read api key from file secret.txt
        with open("secret.txt", "r") as file:
            api_key = file.read()
        self.client = OpenAI(api_key=api_key)
        return

    def text2image_diffusion(self, prompt, num_steps=5, resolution=256):
        """
        This function takes a prompt as input and returns an image as output.
        :param prompt: a string of text
        :return: an image
        """

        if self.pipe is None:
            self.init_diffusion()

        generator = torch.Generator().manual_seed(42)
        generated_image = self.pipe(
            prompt,
            height=resolution,
            width=resolution,
            num_inference_steps=num_steps,
            generator=generator,
        ).images[0]
        return generated_image

    def save_image(self, image, path):
        """
        This function takes an image as input and saves it to a file.
        :param image: an image
        :return: None
        """
        # save the image
        image.save(path)
        return

    def text2image_openai(self, prompt, resolution="1024x1024"):
        """
        This function takes a prompt as input and returns an image as output.
        :param prompt: a string of text
        :return: an image
        """

        if self.client is None:
            self.init_openai()
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=resolution,
                quality="standard",
                n=1,
            )
            return response.data[0].url
        except openai.BadRequestError as e:
            print(e)
            return None

    def save_image_openai(self, image, path):
        """
        This function takes an image as input and saves it to a file.
        :param image: an image
        :return: None
        """
        # get response
        response = requests.get(image, stream=True)

        # save response as png
        with open(path, "wb") as file:
            file.write(response.content)

        # save the image
        # img = Image.open(response.raw)

        # save the image

    def generateImages(self, prompt_path: str, outpath: str):
        """This function takes a list of prompts as input and returns a list of images as output"""
        # generate the images
        # open a json file and load the content
        processor = PreProcessor()

        # load the content
        content = processor.load_json(prompt_path)

        with tqdm(total=len(content), desc="Generating Images...") as pbar:
            for key, value in content.items():
                prompt = value
                # generate the image
                test_image = self.text2image_openai(prompt, "1792x1024")
                if test_image is not None:
                    # save the image
                    
                    self.save_image_openai(test_image, os.path.join(os.path.dirname(__file__), ".",
                                                                    "media", "outputs", outpath, key + ".png"))
                pbar.update(1)

        return
