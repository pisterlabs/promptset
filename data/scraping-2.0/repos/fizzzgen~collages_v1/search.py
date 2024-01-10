import os
import re
import openai
import random
import pprint

from PIL import Image, ImageEnhance

openai.api_key = os.environ['OPENAI_TOKEN']

BACKGROUNDS_PATH = "backgrounds/"
GRAPHICS_PATH = "additional graphics/"
OBJECTS_PATH = "elements/"


class Tokens(object):
    BACKGROUND = "background"
    GRAPHICS = "graphics"
    OBJECTS = "objects"
    DELIMITER = "_"


def _is_image(name):
    return name.lower().endswith((".jpg", ".jpeg", ".png"))


def _tokenize_images(path, main_token):
        result = {}
        for root, _, files in os.walk(path):
            images = [f for f in files if _is_image(f)]
            for image in images:
                tokens = Tokens.DELIMITER.join([main_token] + [
                    token.lower()
                    for token in re.sub('[^0-9a-zA-Z]+', ' ', image.split(".")[0]).split()
                    if token
                ])
                result[os.path.join(root, image)] = tokens
        return result


def gpt_query(query):
    return openai.Completion.create(
        engine="text-davinci-003",
        prompt=query,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.6,
    ).choices[0]["text"].strip().lower()


class ImageSearch(object):
    def __init__(
        self,
        backgrounds_path=BACKGROUNDS_PATH,
        graphics_path=GRAPHICS_PATH,
        objects_path=OBJECTS_PATH,
    ):
        self.images_2_tokens = {}
        # TODO: add download from google drive

        self.images_2_tokens.update(_tokenize_images(backgrounds_path, Tokens.BACKGROUND))
        self.images_2_tokens.update(_tokenize_images(graphics_path, Tokens.GRAPHICS))
        self.images_2_tokens.update(_tokenize_images(objects_path, Tokens.OBJECTS))

    def search(
        self,
        query,
        background_count=1,
        graphics_count=6,
        objects_count=3,
    ):
        query = f"Please select {background_count} background, \
        {graphics_count} graphics, {objects_count} \
        objects for query '{query}' from this set: {self.images_2_tokens.values()}"
        query_result = gpt_query(query)

        result_images = [path for path in self.images_2_tokens.keys() if self.images_2_tokens[path] in query_result]

        result = {
            Tokens.BACKGROUND: [p for p in result_images if self.images_2_tokens[p].startswith(Tokens.BACKGROUND)],
            Tokens.GRAPHICS: [p for p in result_images if self.images_2_tokens[p].startswith(Tokens.GRAPHICS)],
            Tokens.OBJECTS: [p for p in result_images if self.images_2_tokens[p].startswith(Tokens.OBJECTS)],
        }
        for token in [Tokens.GRAPHICS, Tokens.OBJECTS]:
            random.shuffle(result[Tokens.GRAPHICS])

        return result
