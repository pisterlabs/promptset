import os
import base64
import errno
import time
import logging

from openai import OpenAI
from pathlib import Path


# Calculate the project root path directly
project_root_path = Path(__file__).resolve().parent.parent.parent
filename = project_root_path / "logs/gpt_utils.log"
logging.basicConfig(level=logging.DEBUG, filename=filename)


class client_chatGPT():
    """This class summarize the utility methods for chatGPT
    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """

    def __init__(self):
        super(client_chatGPT, self).__init__()
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise Exception("Missing OPENAI_API_KEY environment variable")
        self.client = OpenAI(api_key=api_key)

    def test_connection(self, model_name):
        stream = self.client.chat.completions.create(
            # model="gpt-3.5-turbo-1106" "gpt-4-vision-preview",
            model=model_name,
            messages=[{"role": "user", "content": "who are you? GPT4 or GPT3?"}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")

    def user_message(self, base64_image):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                    {"type": "text", "text": "请用中文回答问题。"}
                ],
            },
        ]

    def analyze_image_with_GPT(self, base64_image, script):
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    # promt produced by chatGPT 13.12.2023
                    "role": "system",
                    "content": """
                        你现在是一个智能助理，专门负责处理和解析图片内容。你的任务包括以下几个方面：
                        图片内容识别：
                            当我提供一张图片时，请详细描述图片中的主要元素，如物体、人物、背景等。
                            尝试捕捉图片的关键细节，例如物体的类型、颜色、人物的表情和活动等。
                        文字识别和解读：
                            识别并解读图片中或周围的任何文字内容。这可能包括标签、说明文字、或图片上的
                            任何注释。
                        回答问题：根据图片内容和任何相关文字，回答我提出的问题。
                        我期望你不仅给出答案，还要解释推导过程和逻辑。
                    """,
                },
            ]
            + script
            + self.user_message(base64_image),
            max_tokens=1000,
        )
        response_text = response.choices[0].message.content
        return response_text

    def encode_image(self, image_path):
        while True:
            try:
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
            except IOError as e:
                if e.errno != errno.EACCES:
                    # Not a "file in use" error, re-raise
                    raise
                # File is being written to, wait a bit and retry
                time.sleep(0.1)

    def load_image(self, path="./artifacts/frames/frame.jpg"):
        # path to your image
        image_path = os.path.join(os.getcwd(), path)

        # getting the base64 encoding
        base64_image = self.encode_image(image_path)

        return base64_image
