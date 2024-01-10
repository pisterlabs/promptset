import openai
from statemachine import StateMachine, State

from api.openai import generate_prompt_based_on_inputs, get_message_from_openai


class RecipeGenerator(StateMachine):
    message = None
    inputs = {}

    ingredients = State("Ingredients", initial=True)
    dietary = State("Diet")
    cuisine = State("Cuisine")
    other = State("Other")
    result = State("Result", final=True)

    ingredients_inserted = ingredients.to(dietary)
    dietary_inserted = dietary.to(cuisine)
    cuisine_inserted = cuisine.to(other)
    other_inserted = other.to(result)

    def insert_inputs(self, inputs: dict):
        if "ingredients" in inputs:
            self.ingredients_inserted()
        if "dietary" in inputs:
            self.dietary_inserted()
        if "cuisine" in inputs:
            self.cuisine_inserted()
        if "other" in inputs:
            self.other_inserted()

    def on_enter_ingredients(self):
        self.message = ("Hi there, I am here to help you find a new recipe.\n\nI'll ask a few questions and then "
                        "present you a recipe. If you don't like it, you can just run the program again.\n\nWhich "
                        "ingredients are you planning to use?")

    def on_enter_dietary(self):
        self.message = ("Got it, I love your taste.\n\nWhat kind of dietary wishes do you have? E.g. vegetarian, "
                        "vegan, lactose-free?")

    def on_enter_cuisine(self):
        self.message = "Alright, I'll keep that in mind.\n\nAny cuisines you like in particular?"

    def on_enter_other(self):
        self.message = ("Noted. I myself am a big fan of Italian, thanks for asking.\n\nAny other things to keep in "
                        "mind? Things like cooking time or amount of persons?")

    def on_enter_result(self):
        self.message = "Great, I'll see if I can do something with this.\n\nYour recipe is getting generated."

    def get_recipe(self, inputs: dict) -> str:
        if "result" in inputs:
            prompt = generate_prompt_based_on_inputs(inputs)
            if prompt:
                try:
                    openai_message = get_message_from_openai(prompt)
                    self.message = "There we go. Your recipe is ready."
                    return openai_message.content
                except openai.AuthenticationError:
                    self.message = "Oops, the admin didn't set the openai key or the key is invalid."
                    return "Generation failed"

            self.message = "Something went wrong during generation, please try again."
            return "Generation failed"
        return ""
