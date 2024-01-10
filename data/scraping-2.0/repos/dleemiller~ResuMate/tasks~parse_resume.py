import logging
import json
from typing import Optional

from api.openai_api import OpenAIMessages, OpenAIMessage, OpenAIRole, ChatCompletionFunction
from api.openai_api import GPT35Turbo as GPTApi
from tasks.dot_logger import DotLogger
from tasks.models.resume import Resume


class ParseResumeFunction(ChatCompletionFunction):
    name = "write_skills"
    description = "Writes the list of skills and experiences."
    param_model = Resume
    gpt_model = GPTApi


class ParseResume:
    """
    Use LLM to parse resume into JSON.
    """

    function = ParseResumeFunction
    temperature = 0.1

    system_prompt = OpenAIMessage(
        role=OpenAIRole.system,
        content="""You will be given an applicant's resume. As an expert recruiter, your task is
        to identify all job skills listed or demonstrated in the document. Carefully evaluate the text
        and extract any information that might interest a potential employer.
        """,
    )

    _user_prompt = OpenAIMessage(
        role=OpenAIRole.user,
        content="""Here is the resume to use for this task:
        {content}
        """,
    )

    @classmethod
    def parse(cls, content: str, *args) -> Resume:
        messages = [
            cls.system_prompt,
            cls._user_prompt.format(content=content),
        ]

        return cls.function.call(messages, temperature=cls.temperature)
