from langchain import LLMChain
from langchain.chains import SequentialChain, TransformChain
from components.lm_connector_factory import LLMConnectorFactory
from templates.website_description_prompt_template import generate_website_description_prompt_template
from transformers.website_context_transformer import transform_website_context


def create_website_description_chain():
    llm_connector = LLMConnectorFactory.create_connector()
    transform_website_context_chain = create_website_context_transformer_chain()

    prompt = generate_website_description_prompt_template()
    llm_chain = LLMChain(
        llm=llm_connector.get_llm,
        prompt=prompt,
        output_key="website_description_summary",
        verbose=True
    )

    return SequentialChain(
        chains=[transform_website_context_chain, llm_chain],
        input_variables=["website_description"],
        output_variables=["website_description_summary"],
        verbose=True
    )


def create_website_context_transformer_chain():
    return TransformChain(
        input_variables=["website_description"],
        output_variables=[
            "website_metadata_url",
            "website_metadata_title",
            "website_metadata_description",
            "website_metadata"
        ],
        transform=transform_website_context
    )
