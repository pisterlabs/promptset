from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from .base_tool import BaseTool


class ResumeTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Resume",
            model="gpt-4",
            temperature=0.75,
            uploads=None,
            inputs=[
                {
                    "input_label": "Job Description",
                    "example": "Write and edit scripts for Youtube videos",
                    "button_label": "Job Keywords",
                    "help_label": "The Resume tool helps by reviewing a given resume and job description and generates an optimized version.",
                },
                {
                    "input_label": "Resume",
                    "example": "Write and edit scripts for Youtube videos",
                    "button_label": "Resume",
                    "help_label": "The Resume tool helps by reviewing a given resume and job description and generates an optimized version.",
                },
            ],
        )

    def execute(self, chat, *inputs):
        instruct_gen_template = f"""\
You are an expert AI job application resume writer and editor.
You apply the following step-by-step process to generate a resume for an applicant:
- You analyze the job description text given to you and generate a list of extracted keywords and potential key points that could be referenced in the applicant's cover letter.
- You analyze the applicant's resume text given to you and generate a list of extracted keywords and key points that relate to the keywords and key points extracted from the job description.
- Then generate an appropriate, concise, and well-written resume that can be shared during application to the job.

Job Description:
---
{inputs[0]}
---

Resume:
---
{inputs[1]}
---

Please respond in this order: 
- The extracted job description keywords and key points
- The relevant extracted resume keywords and key points
- The generated resume

Format your response in markdown.
    """

        user_prompt = HumanMessagePromptTemplate.from_template(
            template=instruct_gen_template
        )
        chat_prompt = ChatPromptTemplate.from_messages([user_prompt])
        formatted_prompt = chat_prompt.format_prompt(
            user_input=inputs[0], user_input_two=inputs[1] if len(inputs) > 1 else None
        ).to_messages()
        llm = chat
        result = llm(formatted_prompt)
        return result.content
