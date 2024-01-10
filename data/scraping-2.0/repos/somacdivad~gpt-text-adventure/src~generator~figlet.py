import json
import logging

from .openai_client import openai, model

game_logger = logging.getLogger("game")
openai_logger = logging.getLogger("openai")

_messages = [
    {
        "role": "system",
        "content": """
I will provide you with a game title and a game description.
You will return the name of a pyfiglet font that you think best fits the title and description.
The font should be small enough to fit on one or two lines.
"""
    },
    {
        "role": "user",
        "content": "TITLE: The Quest for the Sacred Heart DESCRIPTION: As a young adventurer, you are tasked with finding the legendary Crystal Heart, a powerful artifact that has been lost for centuries. With the help of your trusty companions, you must navigate treacherous terrain, battle fierce monsters, and solve challenging puzzles to uncover the secrets of the Crystal Heart and save the realm from darkness."
    },
    {
        "role": "assistant",
        "content": "slant"},
    {
        "role": "user",
        "content": "TITLE: Aria and the Sorcerer’s Curse DESCRIPTION: As the young adventurer Aria, journey through the enchanted forest to rescue your sister from a powerful sorcerer. Battle mythical creatures, solve challenging puzzles, and forge alliances in your quest to save your family and the enchanted forest itself."
    },
    {
        "role": "assistant",
        "content": "epic"},
    {
        "role": "user",
        "content": "TITLE: Whispers in the Shadows DESCRIPTION: As the skilled investigator Marlowe, delve into a world of secrets and intrigue in a city filled with corruption. Unravel a web of lies, use your cunning to outsmart your enemies, and uncover the truth behind a series of mysterious disappearances that threaten to destabilize the city."
    },
    {
        "role": "assistant",
        "content": "ogre"
    }
]


def generate_figlet_font(title: str, description: str):
    game_logger.debug("Generating figlet font")
    response = openai.ChatCompletion.create(
        model=model,
        messages=_messages + [{"role": "user", "content": f"TITLE: {title} DESCRIPTION: {description}"}],
    )
    openai_logger.debug(response)
    font_name = response['choices'][0]['message']['content']
    return font_name
