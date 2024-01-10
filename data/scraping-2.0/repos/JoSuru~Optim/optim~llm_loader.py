from hydra import initialize, compose
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.llms import OpenAI

# Load the configuration
with initialize(config_path="../conf", version_base="1.3"):
    cfg = compose(config_name="config.yaml")


def main():
    """
    The main function loads a Language Model (LLM) from a configuration file and returns the LLM object.
    :return: The LLM object
    """
    # Load the LLM
    if cfg.llm.name == "ollama":
        llm = Ollama(
            model=cfg.llm.model_name,
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        # Return the LLM
        return llm
    if cfg.llm.name == "openai":
        # Load the OpenAI
        return openai_call()


def openai_call():
    """
    This function loads the OpenAI API key and organization from the user's input and sets the environment variables.
    It then returns an OpenAI object.
    :return: The OpenAI object
    """
    return OpenAI(
        openai_api_key=cfg.llm.openai_api_key,
        openai_organization=cfg.llm.openai_organization_id,
    )
