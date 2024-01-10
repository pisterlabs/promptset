#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://oai-generated.openai.azure.com/"
openai.api_version = "2023-06-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

class Oai_create_image:

    def get_response(self, element):

        prompt = element # 良さげな画像になるようにpromptの文章をそのうち考える

        try:
            response = openai.Image.create(
                prompt = prompt,
                size = '1024x1024',
                n = 1
            )
            image_url = response["data"][0]["url"]

            return image_url
        
        except Exception as e:  
            print(f"Error: {e}")
            if e == "Your task failed as a result of our safety system.":
                return "discription error"
            else:
                return "error!"