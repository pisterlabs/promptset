from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate)

def create_translation_prompt(target_language, source_code):
    """
    Create a chat prompt for a code translation task.

    Parameters:
    - target_language (str): The language to which the code should be translated.
    - source_code (str): The source code that needs to be translated.

    Returns:
    langchain.chat_models.ChatPromptTemplate: The generated chat prompt template.
    """
    system_template = "You are a code translator. Your task is to translate the given source code to {target_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "Please translate the following source code to {target_language}: '{source_code}'."
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    return chat_prompt