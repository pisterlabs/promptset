from langchain.chat_models import ChatOpenAI


def initialize_chat_model(api_key: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0) -> ChatOpenAI:
    """
    Initializes a ChatOpenAI model with the specified model name and temperature.

    Args:
        api_key (str): The API key for OpenAI.
        model_name (str): The name of the model to be used. Defaults to "gpt-3.5-turbo".
        temperature (float): The temperature setting for the model. Defaults to 0.

    Returns:
        ChatOpenAI: An instance of ChatOpenAI initialized with the specified model and temperature.
    """
    # Initialize and return the ChatOpenAI model with the specified parameters
    return ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key)


