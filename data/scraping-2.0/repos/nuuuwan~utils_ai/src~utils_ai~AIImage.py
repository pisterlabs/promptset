import time

import openai
from utils_base import Log
from utils_www import WWW

log = Log('AIDraw')


class AIImage:
    DEFAULT_N_IMAGES = 1
    DEFAULT_SIZE = '1024x1024'

    def draw(self, description: str, image_path: str) -> str:
        tic = time.perf_counter()
        response = openai.Image.create(
            prompt=description,
            n=AIImage.DEFAULT_N_IMAGES,
            size=AIImage.DEFAULT_SIZE,
        )
        image_url = response['data'][0]['url']
        WWW.download_binary(image_url, image_path)
        toc = time.perf_counter()
        log.info(
            f'Downloaded "{description}" to {image_path} ({toc - tic:0.4f}s)'
        )
