import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

class dalleIMG:

    def get_dalle(self, prompt):
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
            )
        
        self.image_url = response['data'][0]['url']
        return self.image_url