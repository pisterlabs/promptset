from langchain.prompts.chat import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)

def create_refactoring_prompt(code_snippet):
    """
    Create a chat prompt for an AI assistant specializing in code refactoring.

    Parameters:
    - code_snippet (str): The code snippet for which refactoring suggestions are requested.

    Returns:
    langchain.chat_models.ChatPromptTemplate: The generated chat prompt template.
    """
    # Define system and human message templates for the AI conversation
    system_template = """You are an AI assistant specialized in code refactoring. Your task is to suggest intelligent refinements and automate the refactoring process for the given code snippet."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """Please suggest intelligent refinements and automate the refactoring process for the following code snippet:

    {code_snippet}"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    return chat_prompt