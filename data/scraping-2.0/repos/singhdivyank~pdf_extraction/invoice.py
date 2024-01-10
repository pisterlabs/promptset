"""
extract all details from invoice using GPT
"""
# required imports
import openai

from PIL import Image

# import from file
from aws_connect import textract_extraction
# import constants
from config import (
    INVOICE_PROMPT, 
    OPENAI_MODELS
)


class ImageExtraction:

    """
    using an OpenAI chat model
    extract required contents 
    """

    def __init__(self, image_file):
        self.img = Image.open(image_file)

    def process_file(self):

        """
        performs extraction using GPT
        """

        file_content = ""
        
        try:
            # read image as binary
            with open(self.img, 'rb') as image:
                file_content = textract_extraction(bytearray(image.read()))

            # get GPT response
            gpt_response = openai.ChatCompletion.create(
                model = OPENAI_MODELS.get('chat_model'),
                messages = [
                    {"role": "system", "content": INVOICE_PROMPT},
                    {"role": "user", "content": file_content}
                ]
            )

            return gpt_response["choices"][0]["message"]["content"]
        except Exception as error:
            print(f"process_invoice :: Exception :: {str(error)}")
