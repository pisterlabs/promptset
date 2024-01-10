import os
import openai


class LargeLanguageModel:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def generate_caption(self, image_description):
        messages = [{"role": "system", "content": "Given a textual description of an image, your purpose is to use that information to generate text for an Instagram caption. You come up with witty captions. You do not reply with anything but the caption of the image once given a description. You must not include any hashtags in your response."}]
        message_str = {"role": "user", "content": image_description}
        messages.append(message_str)
        completions = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301", messages=messages
        )
        response = completions["choices"][0]["message"]["content"]
        return response
