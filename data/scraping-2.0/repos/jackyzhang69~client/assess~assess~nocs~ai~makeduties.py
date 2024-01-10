from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
)


def get_duty(position, duty, industry, region):
    template = """
    You are an export of human resource. You need to help generate specific job duties for a given position, based on the generic description from the Canadian National Occupational Classification (NOC) code.  Please convert the following NOC generic descriptions into specific job duties, using the provided example as a guide, and ensure that the duties are appropriate for the specific industry.While maintaining semantic consistency, try to use different words than those in the original description. Additionally, if the original description is short and abstract, consider adding some concrete details to expand it, but avoid generating excessively long content。When referring to nouns mentioned in the NOC generic description, consider whether they are applicable to the specific job, industry, and region, and avoid simply copying from the original text:

    Example:
    Context: This position is marketing manager in restaurant industry in Toronto, Canada. 
    NOC Generic Description: Plan, direct and evaluate the activities of firms and departments that develop and implement advertising campaigns to promote the sales of products and services.
    Specific Job Duty: Strategize, oversee, and assess the initiatives of teams and departments responsible for creating and executing marketing campaigns to increase sales and promote restaurant offerings and services.

    Context: This position is marketing manager in restaurant industry in Toronto, Canada. 
    NOC Generic Description: Establish distribution networks for products and services, initiate market research studies and analyze their findings, assist in product development, and direct and evaluate the marketing strategies of establishments.
    Specific Job Duty: Develop and manage channels to promote menu items and services, conduct market research to identify customer preferences, assist in menu development, and oversee marketing strategies to improve restaurant visibility and sales.

    Context: The position is {position} in the {industry} industry, located in {region}.
    NOC Generic Description: {duty}

    Specific Job Duty:

    """

    prompt = PromptTemplate(
        input_variables=["position", "duty", "industry", "region"],
        template=template,
    )
    pmt = prompt.format(position=position, duty=duty, industry=industry, region=region)
    llm = OpenAI(temperature=0, verbose=False)  # type: ignore
    generated_duty: str = llm(pmt)
    return generated_duty.strip()


def refine(duties):
    template = """
    Revise and reorganize a provided list of job duties, ensuring the removal of repetitive content while maintaining the meaning and intent of each responsibility. The revised list should be concise, effective, and clearly communicate all responsibilities without requiring the preservation of the original order. Present the refined duties without any additional information, decorations, or formatting elements such but not limited to number, dash...
    The duties are:
    {duties}
    """
    prompt = PromptTemplate(
        input_variables=["duties"],
        template=template,
    )
    pmt = prompt.format(duties=duties)

    llm = ChatOpenAI(temperature=0, verbose=False, model_name="gpt-4")  # type: ignore
    result = llm([HumanMessage(content=pmt)]).content
    # clean data
    result_list = result.split("\n")
    result_list = [r for r in result_list if r]
    return result_list
