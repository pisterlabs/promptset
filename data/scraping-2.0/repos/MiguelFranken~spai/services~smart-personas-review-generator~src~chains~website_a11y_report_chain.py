from langchain import LLMChain
from langchain.chains import SequentialChain, TransformChain
from components.lm_connector_factory import LLMConnectorFactory
from templates.website_a11y_report_prompt_template import generate_website_a11y_report_prompt_template
from transformers.website_a11y_report_transformer import transform_website_a11y_report


def create_website_a11y_report_chain():
    llm_connector = LLMConnectorFactory.create_connector()

    transform_website_a11y_report_chain = create_website_a11y_report_transformer_chain()

    prompt = generate_website_a11y_report_prompt_template()
    llm_chain = LLMChain(
        llm=llm_connector.get_llm,
        prompt=prompt,
        output_key="website_a11y_report",
        verbose=True
    )

    return SequentialChain(
        chains=[transform_website_a11y_report_chain, llm_chain],
        input_variables=["accessibility_assessment"],
        output_variables=["website_a11y_report"],
        verbose=True
    )


def create_website_a11y_report_transformer_chain():
    return TransformChain(
        input_variables=["accessibility_assessment"],
        output_variables=[
            "a11y_extended"
        ],
        transform=transform_website_a11y_report
    )
