#
#  dalle2_image_generator.py
#
#  Created by Eldar Eliav on 2023/05/11.
#

import urllib.request
import openai

class Dalle2ImageGenerator:
    class UnsupportedSizeException(Exception):
        pass

    def generate_image_and_download(
        self,
        image_file_destination_with_extension: str,
        prompt: str,
        size: int = 1024
    ):
        if size not in [256, 512, 1024]:
            raise Dalle2ImageGenerator.UnsupportedSizeException(size)

        response = openai.Image.create(
            prompt = prompt,
            n = 1,
            size = f"{size}x{size}"
        )

        image_url = response['data'][0]['url']
        urllib.request.urlretrieve(image_url, image_file_destination_with_extension)
