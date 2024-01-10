import json

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from career_tool.utils.llm_utils import get_completion


class CoverLetter:
    def __init__(self, resume, jd):
        self.resume = resume
        self.jd = jd
        self.cover_letter = self.get_cover_letter()

    def get_cover_letter(self):
        content = (
            f"Given that my resume is: <{self.resume.content}> \n\n"
            f"and job description I am applying is <{self.jd.content}>.\n\n"
            "---\n\n"
            "Can you write me a cover letter? "
            "Don't use flowery language. Be professional."
            # "Don't be very generic. Be specific."
            "Try to keep the tone similar to the resume"
            "Limit the number of words to 500."
        )
        return get_completion(content)

    def get_cover_letter_v2(self, llm_model="gpt-3.5-turbo"):
        chat = ChatOpenAI(temperature=0.0, model=llm_model)

        template_string = """Given the following ```{job_description}```
            and resume ```{resume}```
            Can you write me a cover letter which should be professional?
            Don't use flowery language. Try to keep the tone similar to the resume
            "Limit the number of words to 500.
            """
        prompt_template = ChatPromptTemplate.from_template(template_string)

        service_messages = prompt_template.format_messages(
            job_description=self.jd.content,
            resume=self.resume.content,
        )
        response = chat(service_messages)
        cover_letter = json.loads(response.content)

        return cover_letter
