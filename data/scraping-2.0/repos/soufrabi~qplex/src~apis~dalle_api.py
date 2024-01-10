import openai
import requests
from PySide6.QtGui import QImage, QPixmap

from src.apis.api_key_controller import ApiKeyController


class DALLEApiController:
    def __init__(self):

        print("DALL-E Api")

        self.api_key_controller = ApiKeyController()

    def isPromptValid(self, prompt_obj):

        print(prompt_obj)
        prompt_string = prompt_obj["prompt"]

        if (len(prompt_string) < 10):
            output_dic = {"status": False, "message": "Error :- \n" + "Prompt too small"}
        else:
            output_dic = {"status": True, "message": "Error :- \n" + "Could not recognize type of error"}

        return output_dic

    def get_response(self, prompt_obj):
        OPENAI_API_KEY = self.api_key_controller.get_apikey()

        # Assuming that API key is valid
        openai.api_key = OPENAI_API_KEY

        check_dic = self.isPromptValid(prompt_obj)

        if check_dic["status"] == False:
            return check_dic

        prompt_string = prompt_obj["prompt"]
        print(prompt_string)

        try:
            response = openai.Image.create(
                prompt=prompt_string,
                n=1,
                size="512x512"
            )
            image_url = response['data'][0]['url']

            # Make a GET request to the image URL and get the image data
            response = requests.get(image_url)
            image_data = response.content

            # Create a QImage object from the image data
            image = QImage.fromData(image_data)

            # Create a QPixmap object from the QImage object
            pixmap = QPixmap.fromImage(image)

            # print(pixmap)

            output_dic = {"status": True, "message": "SUCCESS", "pixmap": pixmap}
        except:
            print("Inside the bare error block")
            output_dic = {"status": False, "message": "Error :- \n" + "Could not recognize type of error"}
            pass

        return output_dic
