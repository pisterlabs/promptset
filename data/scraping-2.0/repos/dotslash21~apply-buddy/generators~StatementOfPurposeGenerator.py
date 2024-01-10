from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate


class StatementOfPurposeGenerator:
    def __init__(self, openai_api_key: str, model="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.2,
            openai_api_key=openai_api_key
        )
        self.system_message_prompt_template = SystemMessagePromptTemplate.from_template(
            """You are a helpful assistant, and I need your expertise in crafting a focused and compelling statement 
            of purpose for my university application. Please consider the following elements:

            1. Program and University: [The program and university to which I am applying]
            2. Career Objectives: [My career goals, both immediate and long-term]
            3. Reason for Choosing the Program: [Why this program fits my interests and career trajectory]
            4. Academic Background: Relevant academic accomplishments and studies.
            5. Professional Experience: [My professional experiences that have prepared me for this field]
            6. Skills and Qualities: [My skills and personal attributes that make me a suitable candidate]
            7. Research Interests: [Optional, how my research interests aligns with the program's offerings]
            8. Contribution to the University: [How I aim to contribute to the university and field]
            9. Word Limit: [Word limit for the statement of purpose]
            
            In crafting this statement, seamlessly weave these elements into a cohesive narrative that reflects my 
            passion, commitment, and readiness for the program. Emphasize how my unique background and experiences 
            align with the program's objectives and the university's ethos. Maintain clarity and authenticity, 
            positioning this program as the key stepping stone in my academic and professional journey."""
        )
        self.human_message_prompt_template = HumanMessagePromptTemplate.from_template(
            """
            Program and University: {program_and_university}
            Career Objectives: {career_objectives}
            Reason for Choosing the Program: {reason_for_choosing_the_program}
            Academic Background: {academic_background}
            Professional Experience: {professional_experience}
            Skills and Qualities: {skills_and_qualities}
            Research Interests: {research_interests}
            Contribution to the University: {contribution_to_the_university}
            Word Limit: {word_limit}
            """
        )

    def generate(self, **kwargs) -> str:
        system_message_prompt = self.system_message_prompt_template.format()
        human_message_prompt = self.human_message_prompt_template.format(**kwargs)

        return self.llm([system_message_prompt, human_message_prompt]).content
