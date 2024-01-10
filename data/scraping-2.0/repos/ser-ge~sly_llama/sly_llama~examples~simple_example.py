# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from typing import List, Optional

from sly_llama import llm_call
from langchain.llms import OpenAI

from pydantic import BaseModel

llm = OpenAI()


# + [markdown]
"""
#### Lets define what the add function does and wrap it in an llm call
"""


# -
@llm_call(llm)
def add(x: str, y: str) -> str:
    """
    calculate {x} + {y}
    only return the number and nothing else
    """


add(1, 2)

add(1, 3) + add(1, 1)


# #### Problem: strings don't add, lets try again but with ints
#


# +
@llm_call(llm)
def add(x: str, y: str) -> int:
    """
    calculate {x} + {y}
    only return the number and nothing else
    """


add(1, 3) + add(1, 1)


# + [markdown]
"""
Lets make a recipe
"""

# +


@llm_call(llm)
def get_recipe(dish: str, units: str) -> str:
    """
    Write a resipe for this {dish}
    Be sure to include all the ingridients in {units} units.


    ingridients: < neccesary ingridients>
    intructions: < the instructions for making the dish>
    vegan : <this value must be one of [True, False] indicating weather the recipe is vegan>

    """


# -
print(get_recipe("jank", "metric"))

# + [markdown]
"""
#### That's great but what if we want to parse the output to a pydantic class

#### Let define the output class and how we want to parse the llm output
"""
# +
from pydantic import BaseModel


class Recipe(BaseModel):
    ingridients: str | List[str]
    instructions: str | List[str]
    vegan: bool

    @classmethod
    def from_llm_output(cls, llm_output: str):
        recipe = {}
        parts = llm_output.casefold().partition("instructions")
        recipe["ingridients"] = (
            parts[0].replace("ingridients", "").replace('[],"', "").strip().split("\n")
        )
        recipe["instructions"] = (
            parts[2].partition("vegan")[0].replace('[],"', "").strip().split("\n")
        )
        recipe["vegan"] = bool(
            parts[2].partition("vegan")[1].replace('[],"\n', "").strip()
        )
        return cls.parse_obj(recipe)


# + [markdown]
"""

#### And ammend the return type
"""


# -
@llm_call(llm)
def get_recipe(dish: str, units: str) -> Recipe:
    """
    Write a resipe for this {dish}
    Be sure to include all the ingridients in {units} units.

    ingridients: < neccesary ingridients>
    intructions: < the instructions for making the dish>
    vegan : <this value must be one of [True, False] indicating weather the recipe is vegan>
    """


recipe = get_recipe("kchapuri", "metric")
recipe.instructions

# + [markdown]
"""
#### Hmm that was a lot of work and looks like we did not do a good job, lets ask it to give us some juicy JSON
"""

# +
from sly_llama import JsonBaseModel


class Recipe(JsonBaseModel):
    ingridients: str | List[str]
    instructions: str | List[str]
    vegan: bool


# + [markdown]
"""
#### Llamas are not so good at json so may be let it learn from its mistakes
"""


# -
@llm_call(llm)
def get_recipe(dish: str, units: str, error_message: str) -> Recipe:
    """
    Write a resipe for this {dish}
    Be sure to include all the ingridients in {units} units.

    You should provide your response in JSON Format

    ingridients: < neccesary ingridients>
    intructions: < the instructions for making the dish>
    vegan : <this value must be one of [True, False] indicating weather the recipe is vegan>

    {error_message}
    """


# +
from sly_llama import LlmException

recipe = None
error_message = ""

while not recipe:
    try:
        recipe = get_recipe("kchapuri", "metric", error_message)

    except LlmException as e:
        error_message = e.message
        print(error_message)
recipe
# -


recipe.ingridients
