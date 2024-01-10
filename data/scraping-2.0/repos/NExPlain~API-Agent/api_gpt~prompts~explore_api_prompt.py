from datetime import datetime
from typing import Literal

from langchain import ConversationChain, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from api_gpt.data_structures.proto.generated.workflow_pb2 import WorkflowData
from api_gpt.nlp.exploration import parse_exploration_response
from api_gpt.nlp.exploration_simple import (
    parse_json_from_string,
    parse_workflow_data_from_json,
)
from api_gpt.prompts.explore_api_prompt_conversational import (
    EXPLORE_API_CONVERSATIONAL_SYSTEM_PROMPT,
    EXPLORE_API_CONVERSATIONAL_USER_PROMPT,
)
from api_gpt.prompts.explore_api_prompt_single_chat import (
    EXPLROE_API_SINGLE_CHAT_PROMPT,
)
from api_gpt.services.openai_request import chatopenai
from api_gpt.services.time_zones import get_current_iso_datetime


def generate_api_exploration_workflow(
    chain: LLMChain,
    user_prompt: str,
    workflow_name: str,
    chat_history: str = "",
    mode: Literal["conversational", "single"] = "single",
) -> WorkflowData | None:
    """
    Generates a workflow data object using a given chain, user prompt, user context, and workflow name.

    Args:
        chain (LLMChain): An instance of LLMChain representing the chain used to generate API exploration.
        user_prompt (str): User prompt used to guide the generation of the API exploration.
        user_context (str): User context used to further specify the generation of the API exploration.
        workflow_name (str): The name of the workflow that will be used to name the returned WorkflowData object.

    Returns:
        WorkflowData | None: The WorkflowData object containing data from the generated API exploration.
        If parsing the response fails, None is returned.
    """
    string_response = generate_api_exploration_response(
        chain, user_prompt, chat_history, mode=mode
    )
    if "NEED_MORE_INFORMATION" in string_response:
        workflow_data = WorkflowData()
        workflow_data.name = string_response
        return workflow_data
    return parse_exploration_response(string_response, workflow_name)


def generate_api_exploration_response(
    chain: LLMChain,
    user_prompt: str,
    chat_history: str,
    mode: Literal["conversational", "single"] = "single",
) -> str:
    try:
        if mode == "conversational":
            return chain.run(
                current_time=get_current_iso_datetime(),
                user_prompt=user_prompt,
                chat_history=chat_history,
            )
        else:
            return chain.run(
                current_time=get_current_iso_datetime(),
                user_prompt=user_prompt,
            )
    except Exception as e:
        # Re-raise the exception after catching it
        raise Exception(
            f"An error occurred while generate_api_exploration_response: {str(e)}"
        )


def get_api_exploration_chain(
    mode: Literal["conversational", "single"] = "single"
) -> LLMChain:
    if mode == "single":
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            EXPLROE_API_SINGLE_CHAT_PROMPT
        )

        _human_template = "Text: ${user_prompt}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(_human_template)
        try:
            chat_prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt]
            )
            chain = LLMChain(llm=chatopenai, prompt=chat_prompt)
            return chain
        except Exception as e:
            # Re-raise the exception after catching it
            raise Exception(
                f"An error occurred while generating the API exploration chain: {str(e)}"
            )
    else:
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            EXPLORE_API_CONVERSATIONAL_SYSTEM_PROMPT
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            EXPLORE_API_CONVERSATIONAL_USER_PROMPT
        )
        try:
            chat_prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt]
            )
            chain = LLMChain(llm=chatopenai, prompt=chat_prompt)
            return chain
        except Exception as e:
            # Re-raise the exception after catching it
            raise Exception(
                f"An error occurred while generating the API exploration chain: {str(e)}"
            )
