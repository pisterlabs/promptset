import openai
import os
from dotenv import load_dotenv
from transformers import pipeline

class TextGeneratorInterface:
  def name(self):
    pass

  def generate(self, prompt: str) -> str:
    pass

class DefaultTextGenerator(TextGeneratorInterface):
  def __init__(self):
    self.generator = pipeline('text-generation')

  def name(self):
    return "default"

  def generate(self, prompt: str) -> str:
    output = self.generator.generate(prompt, max_length=800)

    return output[0]['generated_text']


class AdaTextGenerator(TextGeneratorInterface):
  def __init__(self):
    load_dotenv()
    openai.api_key = os.environ['OPENAI_API_KEY']

  def name(self):
    return "ada"

  def generate(self, prompt: str) -> str:
    print('prompt length:', len(prompt))
    output = openai.Completion.create(engine="ada", prompt=prompt, max_tokens=300)

    print(output)
    return output['choices'][0]['text']


class DavinciTextGenerator(TextGeneratorInterface):
  def __init__(self):
    load_dotenv()
    openai.api_key = os.environ['OPENAI_API_KEY']

  def name(self):
    return "da-vinci"

  def generate(self, prompt: str) -> str:
    print('prompt length:', len(prompt))
    output = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=300)

    print(output)
    return output['choices'][0]['text']


