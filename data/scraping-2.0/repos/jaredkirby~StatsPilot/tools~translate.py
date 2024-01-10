"""
Statistical Language Translator: 
A tool to convert mathematical expressions of statistical concepts into plain English and vice versa. 
This would help students with understanding complex mathematical notation and expressing their understanding 
in mathematical terms.
"""

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)


def get_expression_to_english(chat, user_input: str, user_context: str):
    sys_template = """\
    You are a linguist specializing in mathematical language translation.
    Your task is to accurately translate a mathematical expression into plain English, given contextual details 
    about the variables or terms involved. Additionally, you should provide an example related to a 
    specific context or situation to illustrate the expression's usage.
        """
    sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    human_template = f"""\
    The expression you're working with is: 
    "{user_input}"
    Please respond in markdown format.
    """
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    ai_template = f"""\
    Assuming "{user_context}", I can translate "{user_input}" into plain English. 
    Additionally, I'll provide an example to help illustrate its usage.
        """
    ai_prompt = AIMessagePromptTemplate.from_template(ai_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            sys_prompt,
            human_prompt,
            ai_prompt,
        ]
    )
    formatted_prompt = chat_prompt.format_prompt(
        user_input=user_input, user_context=user_context
    ).to_messages()
    llm = chat
    result = llm(formatted_prompt)
    return result.content
