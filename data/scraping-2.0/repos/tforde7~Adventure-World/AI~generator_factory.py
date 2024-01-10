from openai import OpenAI
from AI.audio_generator import AudioGenerator
from AI.image_generator import ImageGenerator
from AI.text_generator import TextGenerator

class GeneratorFactory():
    # This key will be revoked after the project is graded
    CLIENT = OpenAI(api_key="")

    @staticmethod
    def get_image_generator():
        return ImageGenerator(GeneratorFactory.CLIENT)

    @staticmethod
    def get_audio_generator():
        return AudioGenerator(GeneratorFactory.CLIENT)

    @staticmethod
    def get_text_genrator():
        return TextGenerator(GeneratorFactory.CLIENT)


