from langchain.prompts.chat import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)

def create_documentation_prompt(code_snippet):
    """
    Create a chat prompt for a code documentation generator.

    Parameters:
    - code_snippet (str): The code snippet for which documentation is to be generated.

    Returns:
    langchain.chat_models.ChatPromptTemplate: The generated chat prompt template.
    """
    # Define system and human message templates for the AI conversation
    system_template = """You are a code documentation generator. Your task is to automatically generate documentation for the given code snippet."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """Please automatically generate documentation for the following code snippet:

    {code_snippet}
    
    And provide the modified code with the generated documentation.
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    return chat_prompt