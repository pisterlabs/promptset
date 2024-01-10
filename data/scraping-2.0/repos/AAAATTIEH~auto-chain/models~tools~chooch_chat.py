
from langchain.tools import BaseTool
from PIL import Image
import json 
import requests
from base64 import b64encode
from io import BytesIO
class ChoochChatTool(BaseTool):
    name = "Image Chat"
    description = "Image Path Finder Tool must be used before using this tool" \
                 "Use this tool when given the path to an image that you would like to chat about" \
                 "The input to this tool should be a comma separated list of strings of length two, representing the image path and the prompt."

    def _run(self, string):
        image_file, prompt = string.split(',')
        try:
            image = Image.open(image_file).convert('RGB')
        except:
            return "Image path must be found"
        
        # Chooch ImageChat-3 model_id
        model_id = "ad420c2a-d565-48eb-b963-a8297a0e4000"


        # download file
        image_file_url = ""
        #urllib.request.urlretrieve(image_file_url, image_file)

        parameters = {}

        # default is True. If a prompt is given only 1 class will be returned and deep_detection will be turned off
        parameters["deep_inference"] = True

        parameters["prompt"] = prompt


        # replace with your own api key
        api_key = "d79bcaa3-e1d0-490a-8014-dfbfaffd9cb3"

        ENCODING = "utf-8"
        # 1. Reading the binary stuff
        # note the 'rb' flag
        # result: Bytes
        IMAGE_NAME = image_file
        image = Image.open(image_file)
        image = image.convert("RGB")
        image = image.resize((500,300))
        #byte_content= image.tobytes()
        # Create an in-memory BytesIO object
        image_bytes_io = BytesIO()

        # Save the image to the BytesIO object
        image.save(image_bytes_io, format="JPEG")

        # Get the byte content from the BytesIO object
        byte_content = image_bytes_io.getvalue()

        #with open(IMAGE_NAME, "rb") as open_file:
        #    byte_content = open_file.read()

        # 2. Base64 encode read data
        # result: bytes (again)
        base64_bytes = b64encode(byte_content)

        # 3. Decode these bytes to text
        # result: string (in utf-8)
        base64_string = base64_bytes.decode(ENCODING)

        payload = {
            "base64str": base64_string,
            "model_id": model_id,
            "parameters": parameters,
        }

        url = "https://apiv2.chooch.ai/predict?api_key={}".format(api_key)
        response = requests.put(url, data=json.dumps(payload))
        json_data = json.loads(response.content)
        return json_data['predictions'][0]['class_title']



    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")





