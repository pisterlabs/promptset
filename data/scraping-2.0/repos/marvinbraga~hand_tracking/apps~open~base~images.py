import openai

from apps.open.base.abstract import AbstractImageOpenAI
from apps.open.base.utils import StopWatch


class ImageCreate(AbstractImageOpenAI):
    def execute(self):
        stopwatch = StopWatch()
        try:
            response = openai.Image.create(
                prompt=self._prompt,
                n=self._number_of_images,
                size=self._size,
                response_format="b64_json",
            )
        finally:
            stopwatch.stop()

        self._save_image(response)
        return self


class ImageVariation(AbstractImageOpenAI):
    def __init__(self, prompt, image_filename, **kwargs):
        super().__init__(prompt, **kwargs)
        self._image_filename = image_filename

    def execute(self):
        stopwatch = StopWatch()
        try:
            response = openai.Image.create_variation(
                image=open(self._image_filename, "rb"),
                n=self._number_of_images,
                size=self._size,
                response_format="b64_json",
            )
        finally:
            stopwatch.stop()

        self._save_image(response)
        return self


class ImageEdit(AbstractImageOpenAI):
    def __init__(self, prompt, image_filename, image_mask, **kwargs):
        super().__init__(prompt, **kwargs)
        self._image_mask = image_mask
        self._image_filename = image_filename

    def execute(self):
        stopwatch = StopWatch()
        try:
            response = openai.Image.create_variation(
                prompt=self._prompt,
                image=open(self._image_filename, "rb"),
                mask=open(self._image_mask, "rb"),
                n=self._number_of_images,
                size=self._size,
                response_format="b64_json",
            )
        finally:
            stopwatch.stop()

        self._save_image(response)
        return self
