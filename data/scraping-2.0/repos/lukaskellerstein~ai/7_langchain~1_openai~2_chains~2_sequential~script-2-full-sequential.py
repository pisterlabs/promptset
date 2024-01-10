from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# Full Sequence Chain
#
# MULTIPLE INPUTS/OUTPUTS !!!
#
# A more general form of sequential chains,
# allowing for **multiple inputs/outputs**.
# ---------------------------


# ---------------------------
# Example: Generate a name of restaurant and its menu
# ---------------------------
llm = OpenAI(temperature=0.6)

# ---------------------------
# Chain 1: Name of Restaurant
# ---------------------------
prompt_template = PromptTemplate(
    input_variables=["cuisine"],
    template="""I want to open restaurant for {cuisine} food. Suggest a perfect name for the restaurant.""",
)
name_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="restaurant_name")

# ---------------------------
# Chain 2: Menu
# ---------------------------
prompt_template = PromptTemplate(
    input_variables=["restaurant_name"],
    template="""Suggest some menu items for {restaurant_name}. Return it as a list.""",
)
menu_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="menu_items")


# ---------------------------
# Sequential Chain
# ---------------------------
my_chain = SequentialChain(
    chains=[name_chain, menu_chain],
    input_variables=["cuisine"],
    output_variables=["restaurant_name", "menu_items"],
)

# returns both the menu and the name.
result = my_chain({"cuisine": "German"})

print(result)
