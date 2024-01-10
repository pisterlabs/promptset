import os

from kani import Kani
from kani.ext.vision import chat_in_terminal_vision
from kani.ext.vision.engines.openai import OpenAIVisionEngine

api_key = os.getenv("OPENAI_API_KEY")

engine = OpenAIVisionEngine(api_key, model="gpt-4-vision-preview", max_tokens=512)
ai = Kani(engine)

if __name__ == "__main__":
    # use `!path/to/file.png` to provide an image to the engine, e.g. `Please describe this image: !kani-logo.png`
    # or use a URL: `Please describe this image: !https://example.com/image.png`
    chat_in_terminal_vision(ai)
