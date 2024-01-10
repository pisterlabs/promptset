import os.path
from openai_cli.completion.functions.image_to_image import i2i_function
from openai_cli.completion.prompts.image_to_image import i2i_prompt
from abcli import file


def test_i2i_function():
    prompt = i2i_prompt(returns="a darker version of the input image")

    func = i2i_function()

    assert func.generate(prompt.create(func.function_name))[0]
