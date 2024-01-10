import logging

import openai


logger = logging.Logger(__name__)

def generate_prompt(animal: str):
    return """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(animal.capitalize())


if __name__ == "__main__":

    animal="rhino"
    print(generate_prompt(animal=animal))

    assert 1 == 2

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=generate_prompt(animal),
    temperature=0.6
    )
    print(response)