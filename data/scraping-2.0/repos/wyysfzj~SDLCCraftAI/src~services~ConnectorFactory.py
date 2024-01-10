# ConnectorFactory.py

from config import load_config
from OpenAIConnector import OpenAIConnector
from AzureOpenAIConnector import AzureOpenAIConnector

def create_connector():
    config = load_config()  # Implement this method to load the JSON config
    selected_llm = config["selectedLLM"]
    llm_config = next((llm for llm in config["LLMs"] if llm["name"] == selected_llm), None)

    if llm_config["name"] == "OpenAI_GPT":
        return OpenAIConnector(llm_config)
    elif llm_config["name"] == "Azure_OpenAI":
        return AzureOpenAIConnector(llm_config)
    else:
        raise ValueError("Unsupported LLM provider")
