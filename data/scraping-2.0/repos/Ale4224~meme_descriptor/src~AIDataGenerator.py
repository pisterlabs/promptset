from typing import List
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import base64
from dotenv import load_dotenv

load_dotenv()


class AIDataGenerator:
    client = OpenAI()

    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": """I want you to act as a python programming teacher and a comedian. 
You will help me create a funny book about programming for beginners, to teach various programming concepts.
Every page of the book will have a meme image, a title, a sarcastic hyper descriptive sentence of the meme and a simple but useful explanation of the meme.
I will give you the image of the meme, you will generate either the title, the sarcastic sentence or the useful explanation, witchever the user asked.
Do not use double quotes or double new lines in your response. Only give the text that is asked.
""",
        },
    ]

    printable_messages = messages

    def __init__(self, image_path=None, image_url=None):
        if not image_path and not image_url:
            raise Exception("You must provide either an image_path or an image_url")
        self.image_path = image_path or image_url
        self.base64_image = self._encode_image(image_path) if image_path else None
        self.image_url = image_url

    def generate_data(self):
        self._add_image_to_messages()
        return {
            "title": self._generate_image_title(),
            "sarcastic_sentence": self._generate_sarcastic_sentence(),
            "useful_explenation": self._generate_useful_explenation(),
            "image_path": self.image_path,
        }

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _add_image_to_messages(self):
        image_url = (
            f"data:image/jpeg;base64,{self.base64_image}"
            if self.base64_image
            else str(self.image_url)
        )
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is the meme image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        )

    def _execute(self, text):
        print(f"Executing: {text}")
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        )
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=self.messages,
            max_tokens=500,
        )

        res = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": res})

        print(f"Response: {res}")

        return res

    def _generate_image_title(self):
        print("Generating title")
        return self._execute("Generate the title of the meme with one sentence")

    def _generate_sarcastic_sentence(self):
        print("Generating sarcastic sentence")
        return self._execute("Generate the sarcastic sentence")

    def _generate_useful_explenation(self):
        print("Generating useful explenation")
        return self._execute("Generate the useful explenation")
