import os
from dotenv import load_dotenv
from langchain.chains import create_extraction_chain, create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI

load_dotenv()

# Retrieving OpenAI API key from environment variables
treasure_key = os.getenv("OPENAI_API_KEY")

# Initializing the language model with specific settings
language_genie = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=treasure_key)


def unearth_insights(textual_matter: str, **extras):
    """
    Unearth insights from given textual matter based on a defined schema.

    This function gracefully transforms complex data into digestible information chunks,
    making it easier for anyone to understand the magic happening behind the scenes.
    """

    # Pydantic schema is utilized for structured output, enhancing readability
    if "schema_pydantic" in extras:
        response = create_extraction_chain_pydantic(
            pydantic_schema=extras["schema_pydantic"], llm=language_genie
        ).run(textual_matter)
        response_as_dict = [item.dict() for item in response]
        return response_as_dict
    else:
        # Default extraction for unstructured or differently structured data
        return create_extraction_chain(schema=extras["schema"], llm=language_genie).run(
            textual_matter
        )
