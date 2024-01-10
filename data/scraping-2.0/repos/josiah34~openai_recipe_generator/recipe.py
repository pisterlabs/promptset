import openai
import os
import re
from dotenv import load_dotenv


class Recipe:
    def __init__(self):
        self.recipe_title = ""
        self.prompt = ""
        self.ingredients_string = ""
        self.response = ""
        # Load OpenAI API key from .env file
        # Change this to your own .env file
        # Or use the commented out lines below to set your API key
        load_dotenv("secrets.env")
        openai.api_key = os.getenv("OPENAI_KEY")
        # openai.api_key = os.environ['OPENAI_KEY']
        # openai.api_key = "sk-<your key here>"

    # Method to add ingredients to recipe object
    def add_ingredient(self, ingredient):
        self.ingredients_string = ingredient

    # Method to create recipe prompt from ingredients
    def create_recipe_prompt(self):
        self.prompt = (
            f"Create a detailed recipe based on only the following ingredients (Assume user has basic cooking ingredients): {self.ingredients_string}\n"
            + f"Additionally, assign a title starting with 'Recipe Title: ' to the recipe."
        )
        return self.prompt

    # Method to create recipe from prompt using OpenAI API
    def create_recipe(self):
        self.response = openai.Completion.create(
            engine="text-davinci-003",
            max_tokens=300,
            prompt=self.create_recipe_prompt(),
            temperature=0.7,
        )

    # Method to get recipe title from response
    def get_recipe_title(self):
        recipe_text = self.response.choices[0].text
        self.recipe_title = (
            re.findall(r"Recipe Title: (.*)", recipe_text, re.MULTILINE)[0]
            .strip()
            .split("Recipe Title: ")[-1]
        )
        return self.recipe_title
