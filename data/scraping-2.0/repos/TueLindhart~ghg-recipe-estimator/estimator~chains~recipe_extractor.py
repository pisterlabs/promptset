from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

from estimator.prompt_templates.recipe_extractor import (
    RECIPE_EXTRACTOR_PROMPT,
    recipe_output_parser,
)


def get_recipe_extractor_chain(verbose: bool = False):
    prompt = PromptTemplate(
        template=RECIPE_EXTRACTOR_PROMPT,
        input_variables=["input"],
        partial_variables={"format_instructions": recipe_output_parser.get_format_instructions()},
        output_parser=recipe_output_parser,
    )

    llm = ChatOpenAI(  # type: ignore
        temperature=0,
    )
    recipe_extractor_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    return recipe_extractor_chain
