from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chains import SequentialChain

import os
os.environ["OPENAI_API_KEY"] = "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"


llm = OpenAI(temperature=0.7)

# Chain 1: Restaurant Name
prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
)

name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

# Chain 2: Menu Items
prompt_template_items = PromptTemplate(
    input_variables=['restaurant_name'],
    template="""Suggest some menu items for {restaurant_name}. Return it as a comma separated string"""
)

food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

#combining chains
restaurant_chain = SequentialChain(
    chains=[name_chain, food_items_chain],
    input_variables=['cuisine'],
    output_variables=['restaurant_name', "menu_items"]
)
data = restaurant_chain({'cuisine':'Miami'})
print("Cuisine:", data['cuisine'])
print("Restaurant Name:", data['restaurant_name'])
print("Menu Items:")
print(data['menu_items'])