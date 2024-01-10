import json
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import opencopilot.utils.logger as logger
from opencopilot.configs.LLM_Configurations import LLMConfigurations
from opencopilot.configs.env import use_human_for_gpt_4
from opencopilot.utils import network
from opencopilot.utils.consumption_tracker import ConsumptionTracker
from opencopilot.utils.exceptions import APIFailureException, UnsupportedAIProviderException
from opencopilot.utils.open_ai import common
from opencopilot.configs.constants import SupportedAIProviders, LLMModelPriority
from langchain.callbacks import get_openai_callback

llm_configs = None


def initialize_configurations():
    global llm_configs
    llm_configs = LLMConfigurations.execute()


def update_configurations(key1, value1, key2=None, value2=None):
    llm_configs[LLMModelPriority.primary_model.value][key1] = value1
    if key2 is not None:
        llm_configs[LLMModelPriority.secondary_model.value][key2] = value2


def extract_json_block(text):
    # Find the first and last curly brace
    start_index = text.find("{")
    end_index = text.rfind("}")

    if start_index != -1 and end_index != -1:
        # Extract the JSON block from the text
        json_block_text = text[start_index:end_index + 1]

        # Parse the JSON block into a Python dictionary
        try:
            json_block_dict = json.loads(json_block_text)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:")
            print(json_block_text)
            print("Exception:", str(e))
            raise APIFailureException("Error parsing JSON.")

        # Print the extracted JSON block
        return json.dumps(json_block_dict, indent=4)
    else:
        raise APIFailureException("No JSON block found in the text.")


def get_llm_model(llm_names):
    model = None
    # Select the first found configured model
    for llm_model in llm_names:
        if llm_model in llm_configs and "ai_provider" in llm_configs[llm_model]:
            model = llm_configs[llm_model]
            break

    if model is None:
        raise UnsupportedAIProviderException("Didn't find configurations of any of the desired models, "
                                             "Please set the configuration of the desired model in the env!")
    return model


def run(messages, llm_names):
    model = get_llm_model(llm_names)
    llm, model_name = get_llm(model)

    logger.system_message(str("Calling LLM-" + model["ai_provider"] + " " + model_name + " with: \n"))
    logger.operator_input(messages)

    if use_human_for_gpt_4 and "gpt-4" in model_name:
        return common.get_gpt_human_input(messages)

    # Convert messages object to langchain messages model - TODO: it is better to use those objects from the beginning
    langchain_messages = []
    for message in messages:
        content = message["content"]
        if message["role"] == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif message["role"] == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif message["role"] == "assistant":
            langchain_messages.append(AIMessage(content=content))

    with get_openai_callback() as cb:
        llm_reply = network.retry(lambda: llm(langchain_messages))
        consumption_tracking = ConsumptionTracker.create_consumption_unit(model_name, cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.successful_requests, cb.total_cost)

    llm_reply_text = llm_reply.content
    return extract_json_block(llm_reply_text), consumption_tracking


def get_llm(model):
    if model["ai_provider"] == SupportedAIProviders.openai.value:
        return ChatOpenAI(
            openai_api_key=model["openai_text_completion_token"],
            model_name=model["openai_text_completion_model_name"],
            temperature=0
        ), model["openai_text_completion_model_name"]
    elif model["ai_provider"] == SupportedAIProviders.azure_openai.value:
        return AzureChatOpenAI(
            openai_api_key=model["azure_openai_text_completion_token"],
            openai_api_version="2023-05-15",
            openai_api_base=model["azure_openai_text_completion_endpoint"],
            deployment_name=model["azure_openai_text_completion_deployment_name"],
            temperature=0
        ), model["azure_openai_text_completion_deployment_name"]
    else:
        raise UnsupportedAIProviderException("Unsupported AI Provider")
