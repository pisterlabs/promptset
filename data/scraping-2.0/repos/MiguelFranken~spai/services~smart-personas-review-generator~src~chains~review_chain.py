from langchain import LLMChain
from langchain.chains import SequentialChain, TransformChain

from chains.persona_chain import create_transform_persona_chain
from chains.website_a11y_report_chain import create_website_a11y_report_chain
from chains.website_description_chain import create_website_description_chain
from components.lm_connector_factory import LLMConnectorFactory
from templates.review_prompt_template import generate_review_prompt_template


def create_review_chain():
    """
    Creates a SequentialChain object that generates a review based on persona, a11y assessment and website information.
    This is the main chain used by the application.

    Returns:
        SequentialChain: A SequentialChain object that generates a review based on the input variables.
    """
    transform_persona_chain = create_transform_persona_chain()
    website_description_chain = create_website_description_chain()
    website_a11y_report = create_website_a11y_report_chain()

    llm_connector = LLMConnectorFactory.create_connector()
    prompt = generate_review_prompt_template()
    llm_chain = LLMChain(
        llm=llm_connector.get_llm,
        prompt=prompt,
        output_key="review",
        verbose=True
    )

    clean_review_chain = TransformChain(
        input_variables=["review"],
        output_variables=["clean_review"],
        transform=lambda inputs: {"clean_review": inputs["review"].strip(' \n')}
    )

    sequential_chain = SequentialChain(
        chains=[
            transform_persona_chain,
            website_description_chain,
            website_a11y_report,
            llm_chain,
            clean_review_chain
        ],
        input_variables=[
            "website_description",
            "accessibility_assessment",
            "persona",
            "paragraph_limit"
        ],
        output_variables=["clean_review"],
        verbose=True
    )

    return sequential_chain
