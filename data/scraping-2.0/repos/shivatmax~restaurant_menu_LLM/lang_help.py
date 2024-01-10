
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv

def generate_name(cuisine, price):
    load_dotenv()  # take environment variables from .env (especially openai api key)

    llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.7)

    name_template = """I want to open restaurant for {cuisine} cuisine in india.Tell me one unique name for my restaurant."""
    name_prompt = PromptTemplate(template=name_template, input_variables=["cuisine"])
    name_chain = LLMChain(llm=llm, prompt=name_prompt, output_key="restaurant_name")

    menu_template = """Suggestions for menu items for {restaurant_name} along with their prices. I want to keep the price range between 100 to {price} indian rupees."""
    menu_prompt = PromptTemplate(template=menu_template, input_variables=["restaurant_name", "price"])
    menu_chain = LLMChain(llm=llm, prompt=menu_prompt, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, menu_chain],
        input_variables=["cuisine", "price"],
        output_variables=["restaurant_name", "menu_items"]
    )

    print()

    response = chain({"cuisine": cuisine, "price": price})
    
    return response

if __name__ == "__main__":
    print(generate_name("Indian", "1000"))