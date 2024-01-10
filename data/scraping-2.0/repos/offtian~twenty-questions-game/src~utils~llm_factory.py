from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from typing import Union


def create_llm(
    openai_api_type: str,
    deployment_name: str,
    api_base: str,
    api_version: str,
    openai_api_key: str,
    temperature: float = 0.0,
) -> Union[AzureChatOpenAI, ChatOpenAI]:
    """
    Initializes and returns a Language Model instance based on the specified API type.

    Args:
        openai_api_type (str): The type of OpenAI API to use ('azure' or 'openai').
        deployment_name (str): The name of the Azure deployment (used if openai_api_type is 'azure').
        api_base (str): The base URL for the API.
        api_version (str): The version of the API to be used.
        openai_api_key (str): The API key for accessing the OpenAI services.
        temperature (float, optional): The temperature setting for the model's responses. Defaults to 0.0.

    Returns:
        Union[AzureChatOpenAI, ChatOpenAI]: An instance of either AzureChatOpenAI or ChatOpenAI based on the specified type.

    Raises:
        ValueError: If the provided API type is not supported.
    """

    # Initialize AzureChatOpenAI if the selected API type is 'azure'
    if openai_api_type == "azure":
        return AzureChatOpenAI(
            azure_deployment=deployment_name,
            base_url=api_base,
            api_version=api_version,
            api_key=openai_api_key,
            temperature=temperature,
        )

    # Initialize ChatOpenAI if the selected API type is 'openai'
    elif openai_api_type == "openai":
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",  # Assuming the use of GPT-3.5 Turbo model
            api_key=openai_api_key,
            temperature=temperature,
        )

    # Raise an error if an unsupported API type is provided
    else:
        raise ValueError("Unsupported API type")
