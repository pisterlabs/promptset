import openai
import json
from config import OPENAI_API_KEY

class RecipeGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def generate(self, recipe):
        prompts = [{"role": "user", "content": json.dumps(recipe)}]
        instruction = {
            "role": "system",
            "content": "Take the recipes information and generate a recipe with a mouthwatering intro and a step by step guide."
        }
        prompts.append(instruction)

        generated_content = openai.ChatCompletion.create(
            model="gpt-4",
            messages=prompts,
            max_tokens=1000
        )
        return generated_content.choices[0].message.content
