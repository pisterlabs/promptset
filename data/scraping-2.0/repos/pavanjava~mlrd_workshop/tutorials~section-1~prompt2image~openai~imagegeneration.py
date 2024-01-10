import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file


class Prompt2Image:

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def image_from_prompt(self, prompt):
        resp = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return resp



if __name__ == "__main__":
    obj = Prompt2Image()

    response = obj.image_from_prompt("draw image of beautiful lady radha with a neon background lighting and she should be depicted with high focus and cooresponding camera aperture of f/1.44")
    print(response)