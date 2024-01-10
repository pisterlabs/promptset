

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

def get_final_script(chat, outline: str, rough_script: str):
    sys_template = f'''\
    Imagine you are a scriptwriter and script editor for a YouTube channel dedicated to empowering and equipping 
    DIY Entrepreneurs - individuals thirsty for knowledge on how to establish and manage their own businesses 
    from the ground up using contemporary tools and technologies. Your channel seeks to provide practical, 
    detailed walkthroughs on business initiation, bootstrapping strategies, self-taught entrepreneurial skills, 
    and insights gleaned from prosperous entrepreneurs across the annals of history.
    '''
    sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    human_template=f'''\
    The following is the original outline for this video idea:
    ---
    {outline}
    ---

    Given this particular subject matter and video outline, please analyse the following rough draft video script:
    ---
    {rough_script}
    ---

    Please use the following best practices when editing the script:
    1. **Audience-Specific Language**: Use simple, relatable language, avoiding complex business jargon.
    2. **Address Challenges**: Directly tackle the issues faced by your audience and offer practical solutions.
    3. **Storytelling**: Engage your viewers with relatable success stories.
    4. **Actionable Tips**: Provide clear, concise, and actionable guidance, with visual demonstrations if possible.
    5. **User-Generated Content**: Incorporate audience questions and suggestions to enhance interactivity.
    6. **Engaging Openings**: Begin your videos with compelling introductions to hold viewers' attention.
    7. **Concise and Organized Content**: Keep scripts clear, well-structured, and to-the-point, dividing complex 
    topics if needed.
    8. **End with Calls-to-Action**: Encourage viewer engagement and suggest actions like liking, commenting, 
    and subscribing.
    9. **Current Trends**: Keep content relevant by staying updated with the latest entrepreneurial trends.
    10. **Continuous Improvement**: Use audience feedback to refine your scripts and content over time, and 
    maintain a positive, motivating tone throughout.

    Please respond with your improved draft of this video script.
    - The script should be a complete script ready to shoot
    - The script should not be less than 2,000 words
    '''
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            sys_prompt, 
            human_prompt, 
        ]
    )
    formatted_prompt = chat_prompt.format_prompt(outline=outline, rough_script=rough_script).to_messages()
    llm = chat
    result = llm(formatted_prompt)
    return result.content