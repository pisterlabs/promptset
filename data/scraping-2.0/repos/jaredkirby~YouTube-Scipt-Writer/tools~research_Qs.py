

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

def get_research_questions(chat, user_input: int):
    sys_template = f'''\
    You are a communication studies major whose knowledge of media production, marketing, and entrepreneurship would 
    best prepare you to perform research and answer questions related to YouTube video production for a DIY Entrepreneurs 
    focused channel.
    '''
    sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    human_template=f'''\
    You've been tasked with developing a list of research questions required to write a video script from the following 
    rough video outline.

    ---
    {user_input}
    ---

    Please respond with only the required research questions to successfully write the video script based on the video 
    description and outline.
    - The questions should be self contained and not need additional context.
    - Split your questions into sections of no more than 3 questions per section.
    '''
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            sys_prompt, 
            human_prompt, 
        ]
    )
    formatted_prompt = chat_prompt.format_prompt(user_input=user_input).to_messages()
    llm = chat
    result = llm(formatted_prompt)
    return result.content