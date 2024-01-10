from typing import Any
from dotenv import load_dotenv

import openai
import os
import json
import numpy as np
import random


class Quizmaster:
    def __init__(self):
        load_dotenv()
        self.openai = openai
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
        self.prompts = json.load(open("../data/prompts.json", "r"))

        # TODO make methods to load these
        self.few_shot_samples = json.load(open("../data/few-shots.json", "r"))
        self.categories = json.load(open("../data/categories.json", "r"))
        self.moods = json.load(open("../data/moods.json", "r"))

        self.current_category = None

    def _load_categories(self, filename="../data/categories.json"):
        """
        Load the predifined categories and subcategories from a json file.
        """
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    categories = json.load(f)
            except json.JSONDecodeError:
                print(f"File could not be decoded as JSON: {filename}")
            except Exception as e:
                print(f"An error occurred: {e}")

        return categories

    def _load_few_shot_samples(self, filename="../data/few-shots.json"):
        """
        Load the few shot samples from a json file.
        The few shots are loaded in order to provide the LLM with some context,
        and to make it easier for the LLM to generate a question in the right format.
        """
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    few_shot_samples = json.load(f)
            except json.JSONDecodeError:
                print(f"File could not be decoded as JSON: {filename}")
            except Exception as e:
                print(f"An error occurred: {e}")

        return few_shot_samples

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.generate_question(*args, **kwds)

    def generate_question(self, category=None, subcategory=None):
        """
        Design a prompt for the LLM to generate a question.
        The response should be in a json format, which is achieved by prompt engineering.
        If category and subcategory are not provided, they will be randomly selected.
        TODO If it is not in the correct format, the LLM will be asked to try again.
        """

        # Select three random few shot samples
        few_shots = dict()
        few_shots = random.sample(self.few_shot_samples, 3)
        few_shots = json.dumps(few_shots)

        # generate a random category and subcategory
        if category is None or subcategory is None:
            category, subcategory = self.get_random_category_and_subcategory()

        self.current_category = (category, subcategory)

        prompt = f"""
        {self.prompts["user"]} \n
        ({category}, {subcategory}) \n
        {few_shots} \n
        Provide a json response below: \n
        """

        print(prompt)

        # TODO lägg in error hantering ifall gpt-3.5 är upptagen
        response = self.openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.prompts["system"]},
                {"role": "user", "content": prompt},
            ],
        )

        # Retrieve the generated content from the API-response
        content = response["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"Generated content could not be decoded as JSON: {content}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_random_category_and_subcategory(self):
        """
        Return a random category and subcategory from the predefined categories.
        TODO this should return a list of categories and subcategories, so the user can pick one.
        """

        # Produce a list of tuples with (category, subcategory) pairs
        category_and_subcategory_tuples = [
            (k, v) for k, values in self.categories[0].items() for v in values
        ]
        # Select a random tuple from the list
        category, subcategory = random.choice(category_and_subcategory_tuples)
        return category, subcategory

    def rationale(self, question, selected_option, answer):
        """
        Går att smycka ut rejält, typ var otrevlig, eller Explain to me as if I was 5 years old.
        """
        mood = random.choice(self.moods)

        prompt = f"""
        I selected \"{selected_option}\" as the answer to the question \"{question}\". 
        The correct answer was {answer}. \n

        {mood["description"]} \n

        Could you explain the reasoning behind the correct answer and shed light on whether my selection was valid or not? \n
        If you use apostrophes, make sure to escape them with a backslash, like this: \\'
        """

        response = self.openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.8,
            max_tokens=150,
        )

        return mood["emoji"] + " " + response.choices[0].text.strip()


if __name__ == "__main__":
    quizmaster = Quizmaster()
    question = quizmaster()
    print(question)
