import os

from openai import OpenAI

from src.prompt.ddl_generator_prompt import DDLGeneratorPrompt
from src.prompt.er_normalization_prompt import ERNormalizationPrompt
from src.util.image_util import ImageUtil
from src.util.log_util import LogUtil


class CustomLLM:
    def __init__(self):
        self.log = LogUtil()
        self.image_util = ImageUtil()
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate_ddl_using_vision(self, image_path):
        data_url = self.image_util.generate_data_url(image_path)

        # data_url = f"data:image/jpeg;base64,{base64_image}"

        prompt = DDLGeneratorPrompt.prepare_prompt(data_url)

        completion = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=prompt,
            max_tokens=4096,
            stream=True,
        )

        full_response = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            # yield full_response

        # print(completion)

        return full_response

    def normalize_erd_using_vision(self, image_path):
        data_url = self.image_util.generate_data_url(image_path)

        prompt = ERNormalizationPrompt.prepare_prompt(data_url)

        completion = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=prompt,
            max_tokens=4096,
            stream=True,
        )

        full_response = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            # yield full_response

        # print(completion)

        return full_response
