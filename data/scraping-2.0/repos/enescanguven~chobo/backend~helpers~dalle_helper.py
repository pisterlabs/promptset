import os
import hashlib
import random

import openai
import logging

import subprocess
COVERS_FOLDER = "covers"


class DallEHelper():
    def __init__(self, api_key: str) -> None:
        openai.api_key = api_key

    def create_image(self, prompt) -> str:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"
        )
        url = response['data'][0]['url']
        fname = hashlib.md5(url.encode()).hexdigest()+".png"
        self.download_files(url, fname)
        return os.path.join(COVERS_FOLDER, fname)

    @staticmethod
    def download_files(url, fname):
        subprocess.call(["wget", url, "-O", os.path.join(COVERS_FOLDER, fname)])
