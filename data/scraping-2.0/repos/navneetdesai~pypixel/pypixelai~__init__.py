"""
This module is an implementation for the PyPixel library.
"""
import ast
import logging
import os
from datetime import datetime

import pyflakes
import requests
from pyflakes import checker

from .constants import BLACKLIST, DOWNLOAD_DIR, END, SECRETS, START
from .exceptions import *
from .models import Cohere, Model, OpenAI
from .prompts import *


class PyPixel:
    """
    PyPixel class that can be used to generate code or images from a prompt.
    """

    debug: bool = False
    retries = 1

    def __init__(self, model, **kwargs):
        """
        Creates a PyPixel object that can be used to generate code
         or images from a prompt.
        :param model: Model to use for generation
        :param kwargs: Additional arguments [Optional] (Ex: debug, retries)
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.model = model

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Entry for the PyPixel object. Generates code from a prompt.
        :param prompt: Prompt in natural language
        :param kwargs: Additional optional arguments
        :return: Generated code as a string
        """
        code = self.generate_code(GenerateCodePrompt(prompt), self.model)
        self.retries -= 1
        if kwargs.get("write_to_file"):
            self.debug and print("Writing to file...")
            self.write_to_file(code, kwargs.get("write_to_file"))
        if kwargs.get("run_code"):
            while self.retries > 0:
                try:
                    if messages := self.check_code(code):
                        print(f"Warning: {messages}")
                    self.run_code(code)
                    break
                except Exception as e:
                    self.debug and print(f"Retry: {self.retries}")
                    code = self.generate_code(
                        FixCodePrompt(prompt, code, e), self.model
                    )
                    self.retries -= 1
            else:
                raise InvalidCodeException(code)

        return code

    def run_code(self, code):
        """
        Runs the generated code.
        :param code: Code to run
        :return: None
        """
        self.debug and print(f"Running code: {code}")
        exec(code, globals(), locals())

    def __repr__(self):
        """
        Representation of the PyPixel object.
        :return: String representation
        """
        return self.__str__()

    def __str__(self):
        """
        String representation of the PyPixel object.
        :return: String representation
        """
        return "PyPixel"

    def generate_code(self, prompt: Prompt, model: Model) -> str:
        """
        Generates code from a prompt using a model.
        :param prompt: Prompt
        :param model: Model supported by pypixel. Currently supported: OpenAI, Cohere, StarCoder
        :return: Generated code as a string
        """
        if not isinstance(model, Model):
            raise InvalidModelException(str(model))

        if not isinstance(prompt, Prompt):
            raise InvalidPromptException(str(prompt))

        response = model.run(prompt)
        return self.extract_code(response)

    def extract_code(self, text):
        """
        Extracts code from the response text.
        :param text: Response text
        :return: Extracted code
        """
        if START not in text or END not in text:
            raise InvalidCodeException(text)

        code = text.split(START)[1].split(END)[0]
        self.debug and print("Extracting code...")
        if words := [word for word in BLACKLIST if word in code]:
            raise DangerousCodeException(text, words)
        return code

    @staticmethod
    def write_to_file(code, file_name):
        """
        Writes code to a file.
        :param code:
        :param file_name:
        :return:
        """
        with open(file_name, "w") as f:
            f.write(code)

    def check_code(self, code):
        """
        Checks code for errors.
        :param code: Code to check
        :return: List of errors
        """
        self.debug and print("Checking code with pyflakes...")
        messages = []
        tree = compile(code, "generated_code", "exec", ast.PyCF_ONLY_AST)
        checker = pyflakes.checker.Checker(tree, "generated_code")
        for warning in checker.messages:
            message = f"{warning.__class__.__name__}: {warning.message % warning.message_args}"
            messages.append(message)
        return messages

    def generate_images(self, prompt, size=None, num_images=None, download=False):
        """
        Generates images from a prompt.
        :param prompt: Prompt in natural language
        :param size: Size of the image. Default: 256x256
        Supported: 256x256, 512x512, 1024x1024
        :param num_images: Number of images to generate
        :param download: Whether to download the images
        :return: List of image URLs

        """
        if not isinstance(self.model, OpenAI):
            raise InvalidModelException(
                "Invalid model for image generation. Only OpenAI supports image generation."
            )
        self.debug and print("Generating images...")
        image_urls = self.model.generate_images(
            GenerateImagePrompt(prompt), size, num_images
        )
        return self.download(download, image_urls)

    def download(self, download, image_urls):
        """
        Downloads images from a list of image URLs.
        :param download: Whether to download the images
        :param image_urls: List of image URLs
        :return: List of image URLs
        """
        if download:
            for index, image_url in enumerate(image_urls):
                self.download_image(image_url, index)
        else:
            logging.warning(
                "Image download is currently disabled. Use download=True to enable. Image URL expires in 60 minutes."
            )
        return image_urls

    def edit_images(self, image, mask, prompt, n=None, size=None, download=False):
        """
        Edits images based on the prompt
        The transparent areas of the mask indicate where the image should be edited,
        and the prompt should describe the full new image, not just the erased area
        :param image: Image to edit
        :param mask: Mask for the image
        :param prompt: Prompt
        :param n: Number of images to generate (1-10)
        :param size: Size of the image. Default: 256x256.
        :param download: Whether to download the images
        :return:
        """
        if not isinstance(self.model, OpenAI):
            raise InvalidModelException(
                "Invalid model for image editing. Only OpenAI supports image editing."
            )
        self.debug and print("Editing images...")
        image_urls = self.model.edit_images(
            prompt=EditImagePrompt(prompt),
            image=image,
            mask=mask,
            num_images=n,
            size=size,
        )
        return self.download(download, image_urls)

    def download_image(self, image_url, index):
        """
        Downloads an image from a URL.
        :param image_url: URL of the image
        :param index: index for naming the image
        :return: None
        """
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        file_name = os.path.join(
            DOWNLOAD_DIR,
            datetime.now().strftime("%Y-%m-%d-%H:%M:%S%f") + f"_{index}.png",
        )
        try:
            self.debug and print("Retrieving image from url...")
            response = requests.get(image_url)
            response.raise_for_status()
            self.debug and print("Downloading from url...")
            with open(file_name, "wb") as file:
                file.write(response.content)

            print(f"Image downloaded successfully: {file_name}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")
