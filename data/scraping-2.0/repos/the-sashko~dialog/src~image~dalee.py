import openai
import requests
from os import getcwd, path, remove
from typing import Union
from settings.settings import Settings
from logger.logger import Logger

class Dalee:
    __IMAGE_SIZE = '1024x1024'
    __IMAGE_FILE_PATH = '%s/data/tmp/image.png'
    __HTTP_STATUS_OK = 200

    __logger = None

    def __init__(self):
        config = Settings().get_open_ai_config()

        openai.api_key = config['api_key']

        self.__logger = Logger()

    def create_image(self, prompt: str) -> Union[str, None]:
        try: # To-Do: add validator
            response = openai.Image.create(
                prompt = prompt,
                n = 1,
                size = self.__IMAGE_SIZE
            )

            return self.__download_file(
                response['data'][0]['url']
            )
        except Exception as exp:
            self.__logger.log_error(exp)

    def __download_file(self, url: str) -> str:
        file_path = self.__IMAGE_FILE_PATH % getcwd()

        response = requests.get(url)

        if path.isfile(file_path):
            remove(file_path)

        if response.status_code != self.__HTTP_STATUS_OK:
            return None

        with open(file_path, 'wb') as buffer_writer:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    buffer_writer.write(chunk)

        return file_path
