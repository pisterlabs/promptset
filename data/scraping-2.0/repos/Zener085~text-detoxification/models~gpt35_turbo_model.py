"""
Detoxication using models developed by OpenAI.
This model is taken because I did not pay even 1$ and I don't have the access to the gpt-4. But you can change it if you
wish.
"""
__author__ = "Zener085"
__version__ = "1.0.0"
__license__ = "MIT"
__all__ = ["preprocess_dataset", "gpt_detox"]

import openai
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame
from random import randint
from os import getenv
from typing import Tuple, List

load_dotenv(find_dotenv())
openai.api_key = getenv("OPENAI_API_KEY")


def preprocess_dataset(__df: DataFrame) -> Tuple[List[str], List[List[str]]]:
    """
    Preprocess dataset to create input and outputs lists for the gpt model. Works only for the main dataset of this
    project.

    Args:
        __df: Dataset that can be preprocessed.

    Returns:
        Input and output lists for the gpt.
    """
    n = 10
    index = randint(n, __df.shape[0] - n)

    _inputs = __df["reference"][index:index + n].to_list()
    _outputs = __df["translation"][index:index + n].to_list()
    for i in range(len(_outputs)):
        _outputs[i] = [_outputs[i]]

    return _inputs, _outputs


def gpt_detox(__text: str, __model: str = "gpt-3.5-turbo", __temperature: float = 0) -> str:
    """
    Asks the model to detox the text.

    Args:
        __text: A text with not low level of toxicity.
        __model: A model that is used to detoxicate the text. The default model is "gpt-3.5-turbo".
        __temperature: A temperature of the model. The default value is 0. I don't suggest increasing this value.

    Returns:
        Detoxicated text.
    """
    _message = [{"role": "system",
                 "content": "You will be provided with a sentence with high level of toxicity, and your task is to "
                            "detoxicate it."},
                {"role": "user",
                 "content": f"{__text}"}
                ]
    _detoxed_text = openai.ChatCompletion.create(model=__model,
                                                 temperature=__temperature,
                                                 messages=_message,
                                                 max_tokens=256
                                                 )
    return _detoxed_text.choices[0].message["content"]


if __name__ == "__main__":
    toxic_text = "monkey, you have to wake up."
    print(gpt_detox(toxic_text))
