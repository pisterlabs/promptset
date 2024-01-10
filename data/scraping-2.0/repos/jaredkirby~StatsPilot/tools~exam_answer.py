from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)


def get_exam_question_answer(chat, user_input: str):
    sys_template = """
    You are a Statistics and data expert taking a statistics exam.
    Your task is to walk through the thought process and calculations for a given statistics exam question 
    and provide a clear final answer. Your response should be formatted in markdown for clarity.
    """
    sys_message_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    question_template = f"""\
    Exam question:
    {user_input}
    Think through your approach step by step, making sure to show all calculations and formulas used. 
    Finally, provide a clear and concise final answer to the exam question.
    Format your response in markdown.
    """
    question_message_prompt = HumanMessagePromptTemplate.from_template(
        question_template
    )
    response_template = "Sure! Let's think step by step:"
    response_message_prompt = AIMessagePromptTemplate.from_template(response_template)
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            sys_message_prompt,
            question_message_prompt,
            response_message_prompt,
        ]
    )
    formatted_answer_prompt = answer_prompt.format_prompt(
        user_input=user_input
    ).to_messages()
    llm = chat
    result = llm(formatted_answer_prompt)
    print("Bot Response:", result.content)
    return result.content
