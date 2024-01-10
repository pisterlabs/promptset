from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import json


# The data source
food_label = '/Users/arshad/Desktop/Projects/Healthy-Eats/sample_image/sample.jpeg'


# Prompt Template
ingredients_template = PromptTemplate(
    input_variables=['food_label'],
    template="""You are a great Ingredient Parser who can extract ingredients from a given food label text.
    Extract the ingredients from the following food_label:
    FOOD LABEL: {food_label}"""
)

template_string = """You are a master ingredient parser from a given food label. You give detailed descriptions of the ingredients\
You can classify each ingredient as Healthy/Unhealthy.
You also add emojis for each ingredient.

Take the Food Label below delimited by triple backticks and use it to extract the ingredients and provide a detailed description.

brand description: ```{food_label}```

then based on the description you give the brand an Emoji and a label for healthy or unhelathy.

Format the output as JSON with the following keys:
Ingredient
Description
Emoji
Healthy/Unhealthy label
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

chat_llm = ChatOpenAI(temperature=0.0)
llm = OpenAI(temperature=0)
ingredients_chain = LLMChain(
    llm=llm, prompt=ingredients_template, verbose=True, output_key='ingredients')

ingredients_list = prompt_template.format_messages(
    food_label=ingredients_chain.run(food_label))

response = chat_llm(ingredients_list)

final_response = response.content

data_dict = json.loads(final_response)



