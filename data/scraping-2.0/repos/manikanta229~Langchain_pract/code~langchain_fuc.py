from langchain.llms import OpenAI
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from keys import Acess_Key
import os
os.environ['OPENAI_API_KEY'] = Acess_Key

llm = OpenAI(temperature=0.6)

 #here we are implementing Sequentialchain where it can take multiple inputs and gives multiple ouputs.
def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for it"
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name,output_key="restaurant_name")

    prompt_template_items = PromptTemplate(
    input_variables=['restaurant_name'],
    template="""Suggest some menu items for {restaurant_name}. Return it as a comma-separated list"""
    )

    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items,output_key="menu_items")
    chain=SequentialChain(
    chains = [name_chain,food_items_chain],
    input_variables = ['cuisine'],
    output_variables = ['restaurant_name','menu_items']
    )
    response = chain({'cuisine':'Arabic'})
    return response