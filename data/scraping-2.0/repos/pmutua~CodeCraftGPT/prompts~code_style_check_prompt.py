from langchain.prompts.chat import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)

def create_coding_style_prompt(refined_code):
    """
    Create a chat prompt for an AI assistant providing feedback on coding style.

    Parameters:
    - refined_code (str): The code for which feedback and suggestions are requested.

    Returns:
    langchain.chat_models.ChatPromptTemplate: The generated chat prompt template.
    """
    # Define system and human message templates for the AI conversation
    system_template = """You are an AI assistant designed to provide real-time feedback on coding style and offer suggestions for improvement."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """Please provide feedback and suggestions for improving the coding style of the following code:

    {refined_code}"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    return chat_prompt