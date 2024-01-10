from langchain.llms import OpenAI, GPT4All
from langchain.chat_models import ChatOpenAI


ALL_MODELS = [
    "chat-openai",
    # "gpt-4",
    # "gpt-4-0314",
    # "gpt-4-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-0301 (Legacy)"
    # "gpt4all-falcon",
    # "gpt4all-13b-snoozy",
    # "meta-llama/Llama-2-70b-chat-hf",
    # "meta-llama/Llama-2-13b-chat-hf",
    # "meta-llama/Llama-2-70b-chat",
    # "meta-llama/Llama-2-13b-chat",

]


def get_llm(model_name: str, temperature: float = 0.0):
    """Create an llm based on model name

    Supported models:
        - chat-openai
        - gpt-3.5-turbo
        - gpt-3.5-turbo-0301
        - gpt-3.5-turbo-0613
        - gpt-3.5-turbo-16k
        - gpt-3.5-turbo-16k-0613
        - gpt4all-falcon
        - gpt4all-13b-snoozy

    Args:
        model_name: The name of the model.
        temperature: The temperature of the model.

    Returns:
        The llm.

    Raises:
        ValueError: If the model name is not supported.

    Examples:
        >>> llm = get_llm("chat-openai")
    """
    if model_name not in ALL_MODELS:
        raise ValueError(
            f"Model name {model_name} is not supported. Please use one of the following: {ALL_MODELS}"
        )

    if model_name.startswith("gpt-3.5"):
        llm = OpenAI(temperature=temperature, model_name=model_name)
    elif model_name == "chat-openai":
        llm = ChatOpenAI()
    elif model_name == "gpt4all-falcon":
        llm = GPT4All(
            model="./models/ggml-model-gpt4all-falcon-q4_0.bin",
            max_tokens=512,
            n_threads=8,
        )
    elif model_name == "gpt4all-13b-snoozy":
        llm = GPT4All(
            model="./models/GPT4All-13B-snoozy.ggmlv3.q4_0.bin",
            max_tokens=512,
            n_threads=8,
        )

    return llm
