from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# Simple Sequence Chain
#
# ONLY ONE INPUT AND ONE OUTPUT !!!!
#
# The simplest form of sequential chains, where each step
# has a singular input/output, and the output of one step
# is the input to the next.
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
name_chain = LLMChain(llm=llm, prompt=prompt_template)

# ---------------------------
# Chain 2: Menu
# ---------------------------
prompt_template = PromptTemplate(
    input_variables=["restaurant_name"],
    template="""Suggest some menu items for {restaurant_name}. Return it as a list.""",
)
menu_chain = LLMChain(llm=llm, prompt=prompt_template)


# ---------------------------
# Simple Sequential Chain
# ---------------------------
my_chain = SimpleSequentialChain(
    chains=[name_chain, menu_chain],
)

# returns only the menu, as the name is not returned.
result = my_chain.run("German")

print(result)
