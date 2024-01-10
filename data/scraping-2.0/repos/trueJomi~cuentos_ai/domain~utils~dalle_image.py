import os
from dotenv import load_dotenv

load_dotenv()


class Dalle2:
    
    def __init__(self):
        import openai
        self.__openai = openai
        self.__openai.api_key= os.getenv("OPENAI_API_KEY")
    
    def generateImage(self,prompt:str)->str:
        response = self.__openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512",
        )
        return response["data"][0]["url"]
