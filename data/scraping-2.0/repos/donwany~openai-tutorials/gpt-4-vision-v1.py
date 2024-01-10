from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class GPTAssistant:
    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_image_description(self, image_url):
        """generate image description"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "What's in this image? Can you also extract the figures in the image and put it in a json format?"},
                    {"type": "image_url", "image_url": image_url}
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=300
        )

        return response.choices[0].message.content


def main():
    api_key = os.getenv("api_key")
    assistant = GPTAssistant(api_key)

    image_url = "https://s3.amazonaws.com/youtube-demo-bkt/Presidential-Results-Sheets-Greater-Accra-34-726x1024.jpg"
    result_content = assistant.generate_image_description(image_url)

    print(result_content)


if __name__ == '__main__':
    main()
