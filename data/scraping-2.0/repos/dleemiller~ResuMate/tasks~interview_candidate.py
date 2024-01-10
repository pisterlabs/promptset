import logging
import json
from typing import Optional

from api.openai_api import OpenAIMessages, OpenAIMessage, OpenAIRole, GPT35Turbo
from tasks.models.resume import Resume

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InterviewCandidate:
    """
    Use LLM to perform interview screening.
    """

    model = GPT35Turbo

    _system_prompt = OpenAIMessage(
        role=OpenAIRole.system,
        content="""
As an expert interviewer, your task is to assess a candidate's suitability for a given job. You will be provided with the candidate's resume and a job listing. Use the resume to formulate questions aimed at determining if the candidate is a good fit for the position. Focus on aspects outlined in the job listing that are not explicitly covered in the resume.

Here are the guidelines:

- Ask one question at a time, each addressing a different area of the job listing.
- Wait for a response from the candidate before asking the next question.
- Your questions should be designed to uncover whether the candidate possesses relevant experience not highlighted in the resume.
- Avoid redundant questions that can be adequately answered by the resume.
- Begin by assessing if the candidate is broadly aligned with the position before delving into specific details.
- Maintain a neutral and analytical tone throughout the interview. Refrain from offering remarks on the candidate's responses.
- If a question pertains specifically to the candidate's experience, incorporate the company name into the question.

To illustrate, you might start with a statement about missing experience and then proceed to inquire further. For instance, "Company X is seeking someone with Y experience; however, I noticed your resume emphasizes Z. Can you provide more insights into your experience related to Y?"

Please keep your questions succinct and to the point. Skip any question if the candidate's resume already adequately addresses the relevant experience or skills.

Here is the job listing to use for this task:
{job_listing}
        """,
    )

    _user_prompt = OpenAIMessage(
        role=OpenAIRole.user,
        content="{answer}",
    )

    _assistant_prompt = OpenAIMessage(
        role=OpenAIRole.assistant,
        content="{question}",
    )

    @classmethod
    def interview(cls, resume: Resume, job_listing: str):
        messages, question = cls.begin_interview(resume, job_listing)
        while True:
            answer = str(input(question))
            if answer.lower() == "exit":
                break
            messages, question = cls.interact(messages, answer)

        with open("interview.json", "w") as fh:
            json.dump(messages.dict(), fh, indent=4)

        logger.info("Successful interview.")

    @classmethod
    def begin_interview(
        cls, resume: Resume, job_listing: str
    ) -> tuple[OpenAIMessages, str]:
        messages = OpenAIMessages(
            messages=[
                cls._system_prompt.format(job_listing=job_listing),
                cls._user_prompt.format(
                    answer=f"Here is my resume, let's begin:\n{resume.short_version()}"
                ),
            ]
        )
        logging.debug(f"messages: {list(messages)}")
        response = cls.model.create(messages)

        question = response.choices[0].message.content
        messages.append(cls._assistant_prompt.format(question=question))
        logger.info(question)
        return messages, question

    @classmethod
    def interact(cls, messages, answer: str) -> tuple[list[dict], str]:
        messages.append(cls._user_prompt.format(answer=answer))
        logging.debug(f"messages: {list(messages)}")
        response = cls.model.create(messages)

        question = response.choices[0].message.content
        logger.info(question)
        messages.append(cls._assistant_prompt.format(question=question))
        return messages, question
