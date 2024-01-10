

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

def get_rough_script(chat, outline: str, research="No Research"):
    sys_template = f'''\
    You are a communication studies major whose knowledge of media production, marketing, and entrepreneurship 
    would best prepare you to write scripts related to YouTube videos for a DIY Entrepreneurs focused channel.
    '''
    sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    human_template=f'''\
    Y**Target Audience Characteristics:**
    The DIY Entrepreneurs audience would be largely made up of individuals who have a strong independent streak 
    and prefer self-guided learning over structured, institutional education. They may be in different phases of 
    their entrepreneurial journey, from those just toying with the idea of starting a business to those who have 
    already started but want to expand their knowledge and skills. This audience may also include freelancers, 
    independent consultants, or solo entrepreneurs who run their own businesses single-handedly.

    **Video Description and Outline Reference:**
    {outline}

    **Video Outline Research Reference:**
    {research}

    **Task:**
    Please write a full and complete YouTube video script written for the stated target audience using the video 
    outline and description provided.
    - The video style will be a voice-over video.
    - Use the supplied research information when writing the script.
    - The script should be 2,000 words.
    - Begin with writing the main content of the video followed by the introduction and conclusion.
    '''
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            sys_prompt, 
            human_prompt, 
        ]
    )
    formatted_prompt = chat_prompt.format_prompt(outline=outline, research=research).to_messages()
    llm = chat
    result = llm(formatted_prompt)
    return result.content