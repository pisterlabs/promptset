from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from loguru import logger


def main():
    """https://python.langchain.com/docs/modules/chains/foundational/sequential_chains"""
    load_dotenv(find_dotenv())

    llm = OpenAI(temperature=0.0, verbose=True)
    template = """
    Food: {food}
    Flavor: {flavor}
    Ingredients:
    """
    prompt = PromptTemplate.from_template(template)
    ingredient_chain = LLMChain(llm=llm, prompt=prompt, output_key='ingredients', verbose=True)

    llm = OpenAI(temperature=0.0)
    template = """
    Food: {food}
    Flavor: {flavor}
    Ingredients: {ingredients}
    Instructions:
    """
    prompt = PromptTemplate.from_template(template)
    instruction_chain = LLMChain(llm=llm, prompt=prompt, output_key='instructions', verbose=True)

    recipe_chain = SequentialChain(chains=[ingredient_chain, instruction_chain],
                                   input_variables=['food', 'flavor'],
                                   output_variables=['instructions'],
                                   verbose=True)
    recipe = recipe_chain.run(dict(food='ice cream', flavor='vanilla'))
    logger.info("recipe: {}", recipe)
    print(type(recipe))


if __name__ == '__main__':
    main()
