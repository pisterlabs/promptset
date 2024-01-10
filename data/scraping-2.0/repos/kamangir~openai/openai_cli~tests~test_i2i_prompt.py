import os.path
from openai_cli.completion.prompts.image_to_image import i2i_prompt
from abcli import file


def test_i2i_prompt():
    prompt = i2i_prompt(
        returns="a darker version of the input image",
    )

    text = prompt.create("abc").replace("\n", "  ")
    while "  " in text:
        text = text.replace("  ", " ")

    assert (
        text
        == "Write a python function named abc that inputs an image as a numpy array and does not run a for loop on the pixels and uses numpy vector functions and imports all modules that are used in the code and type-casts the output correctly and returns a darker version of the input image as a numpy array."
    )
