from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def recipe_chain(llm, system_template, human_template):
    first_system_template = SystemMessagePromptTemplate.from_template(system_template)
    first_human_template = HumanMessagePromptTemplate.from_template(human_template)
    first_prompt = ChatPromptTemplate.from_messages([first_system_template, first_human_template])
    # Chain 1
    recipe_chain = LLMChain(llm=llm, prompt=first_prompt, output_key="recipe")
    return recipe_chain


def ingredients_chain(llm, template):
    second_prompt = ChatPromptTemplate.from_template(template)

    ingredients_chain = LLMChain(
        llm=llm, prompt=second_prompt, output_key="ingredients"
    )

    return ingredients_chain


def chain(recipe_chain, ingredient_chain):
    overall_chain = SequentialChain(
        chains=[recipe_chain, ingredient_chain],
        verbose=True,
        input_variables=["food", "format_instructions", "example_instructions"],
        output_variables=["recipe", "ingredients"],
    )

    return overall_chain
