import openai

from .config import MAX_TOKENS, MODEL, TEMPERATURE


class TextClient(openai.Completion):

    def _get_response(self, text) -> dict:
        response = self.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            prompt=text,
        )
        return response

    def get_text(self, *args) -> str:
        response = self._get_response(*args)
        text = response['choices'][0]['text']
        return text.strip().replace('\n', ' ')


class ImageClient(openai.Image):

    def _get_response(self, text) -> dict:
        response = self.create(prompt=text, n=1)
        return response

    def get_image_url(self, *args) -> str:
        response = self._get_response(*args)
        image_url = response['data'][0]['url']
        return image_url
