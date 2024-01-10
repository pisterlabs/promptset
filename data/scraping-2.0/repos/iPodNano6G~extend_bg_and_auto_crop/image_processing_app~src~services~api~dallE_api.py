from config import dallE_api_key

import openai
import cv2

from utils.files.file_downloader import FileDownloader
from utils.image.image_writer import ImageWriter

class DallEAPI():
    def outpainting(self, np_image, length = 1024) -> 'numpy.ndarray': 
        openai.api_key = dallE_api_key

        np_image = cv2.resize(np_image, (length, length))

        cv2.imwrite("outpainting_temp.png", np_image)
        outpainted = openai.Image.create_edit(
        image = open("outpainting_temp.png", "rb"),
            prompt="photo of person",
            n=1,
            size= str(length)+"x"+str(length)
        )
        image_url = outpainted.data[0]['url']

        dallE_image_path = FileDownloader().download_image(image_url, file_name = "dallE.png")
        dallE_np_image = cv2.imread(dallE_image_path)

        return dallE_np_image