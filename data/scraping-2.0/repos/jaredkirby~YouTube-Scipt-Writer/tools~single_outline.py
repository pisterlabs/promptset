

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

def get_outline(chat, selected_input: str):
    sys_template = f'''\
    You are a digital media and communication specialist who is well-versed in content creation strategies and techniques 
    for YouTube videos and can provide informed guidance and actionable advice to DIY entrepreneurs.
    '''
    sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    human_template=f'''\
    From the following are variations of 3 YouTube video descriptions and outlines for a given topic: 

    --- 
    {selected_input}
    ---

    Extract and combine the best ideas that adhere to the following principles: 

    To create compelling and engaging content, particularly in the realm of DIY entrepreneurship, consider the 
    following guidelines:

    1. **Information Visualization:** Make use of infographics, charts, diagrams, and other visual aids to explain 
    complex ideas effectively. 

    2. **Captivating Start:** Grab your audience's attention from the outset with intriguing facts, compelling 
    questions, or strong statements.

    3. **Personal Stories:** Engage your audience emotionally by incorporating personal experiences or field-related 
    stories for relatability. 

    4. **Narrative Arc:** Create a story-like structure in your content, including a beginning, middle, and end that 
    present situations, challenges, and outcomes to maintain engagement. 

    5. **Use of Metaphors and Analogies:** Simplify complex concepts with metaphors and analogies, making them more 
    understandable and relatable.

    6. **Conflict and Resolution:** Discuss relevant challenges in DIY entrepreneurship and provide solutions, tips, 
    and advice to overcome them. 

    7. **Case Studies:** Include examples of successful DIY entrepreneurs to demonstrate proof of concept and enable 
    your audience to learn from these real-world experiences.

    8. **Emotional Connection:** Evoke emotional responses in your audience through humor, passion, excitement, or 
    shared challenges. 

    9. **Clarity and Simplicity:** Ensure your content is easy to understand by using clear and concise language and 
    avoiding jargon. 

    10. **Audience Interaction:** Encourage active participation with interactive elements such as calls to action and 
    invites for comments, likes, subscriptions, or shares. 

    11. **High-Quality Production:** Use high-quality visuals and sound to improve engagement and give your content a 
    professional touch. 

    12. **Consistent Style:** Keep a consistent style or theme throughout your content to aid in recognition and build 
    trust with your audience.

    13. **Key Points Reinforcement:** Reiterate important concepts using different methods to strengthen audience 
    comprehension. 

    14. **Anticipation Building:** Include cliffhangers or teasers at the end of each video to create anticipation 
    for future content, keeping your audience hooked.

    15. **Audience Involvement:** Engage your audience further by featuring viewer comments, hosting Q&A sessions, 
    or inviting user submissions.

    16. **Influencer or Expert Testimonies:** Add credibility and inspiration to your content by incorporating insights 
    or testimonials from respected figures in entrepreneurship.

    Respond with a single video description and outline that is the best combination of the provided ideas.
    '''
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            sys_prompt, 
            human_prompt, 
        ]
    )
    formatted_prompt = chat_prompt.format_prompt(selected_input=selected_input).to_messages()
    llm = chat
    result = llm(formatted_prompt)
    return result.content