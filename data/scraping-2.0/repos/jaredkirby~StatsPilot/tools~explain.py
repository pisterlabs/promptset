from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def get_explanation(chat, user_input: str, level: int):
    sys_template = f"""\
    You are a helpful Statistics expert assisting a student further their understanding.
    Your task is to explain "{user_input}" in a step-by-step manner, using practical examples to help illustrate its applications. 
    The explanation should be tailored to a {level} grade level and should be broken down into smaller, digestible parts to make it easier to understand. 
    If possible, please also recommend any additional resources, such as textbooks, websites, or online courses, 
    that may deepen the user's understanding of "{user_input}".
    """
    sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    human_template = f"""\
    Could you please explain "{user_input}" in a step-by-step manner, using practical examples to help illustrate its applications? 
    Please tailor your explanation to a {level} grade level and break it down into smaller, digestible parts. 
    Additionally, if possible, could you please recommend any further resources that may deepen my understanding of {user_input}?
    """
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            sys_prompt,
            human_prompt,
        ]
    )
    formatted_prompt = chat_prompt.format_prompt(
        user_input=user_input, level=level
    ).to_messages()
    llm = chat
    result = llm(formatted_prompt)
    print("Bot Response:", result.content)
    return result.content
