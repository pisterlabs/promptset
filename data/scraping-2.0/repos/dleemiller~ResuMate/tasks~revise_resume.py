import logging
import json
from typing import Optional

from api.openai_api import OpenAIMessages, OpenAIMessage, OpenAIRole, GPT35Turbo
from tasks.models.resume import Resume

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def format_interview_qa(interview_qa) -> str:
    interview_started = False
    qa_text = ""
    for q in interview_qa["messages"]:
        if q["role"] == OpenAIRole.system.value:
            continue
        elif q["role"] == OpenAIRole.assistant.value:
            interview_started = True
        elif not interview_started:
            continue

        if q["role"] == OpenAIRole.user.value:
            qa_text += f"Candidate: {q['content']}\n"
        elif q["role"] == OpenAIRole.assistant.value:
            qa_text += f"Interviewer: {q['content']}\n"

    return qa_text


class ReviseResume:
    """
    Use LLM to perform resume revision.
    """

    model = GPT35Turbo

    _system_prompt = OpenAIMessage(
        role=OpenAIRole.system,
        content="""
You are an expert at writing resumes, with deep knowledge of the candidate's field.
You will be given a resume, a job listing and a question and answer session between an interviewer and the candidate.

Your task is to rewrite the resume to appeal to the job listing.
You can use information from the original resume, as well as the interview question and answer to help improve the resume for the job listing.
Finally, the candidate will provide feedback for you to incorporate.

You will only respond with the rewritten resume and nothing else.

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
        content="{revision}",
    )

    @classmethod
    def write(cls, resume: Resume, interview_qa: list[dict], job_listing: str):
        messages = cls.begin_writing(resume, interview_qa, job_listing)
        while True:
            answer = str(input("Instruct the llm to make any revisions: "))
            if answer.lower() == "exit":
                break
            messages = cls.interact(messages, answer)

        with open("revised_resume.json", "w") as fh:
            json.dump(messages.dict(), fh, indent=4)

        logger.info("Success.")

    @classmethod
    def begin_writing(
        cls, resume: Resume, interview_qa: list[dict], job_listing: str
    ) -> tuple[OpenAIMessages, str]:
        qa_text = format_interview_qa(interview_qa)
        messages = OpenAIMessages(
            messages=[
                cls._system_prompt.format(job_listing=job_listing),
                cls._user_prompt.format(
                    answer=f"Here is my conversation with the interviewer:\n{qa_text}\n\nHere is my resume, let's begin:\n{resume.short_version()}"
                ),
            ]
        )
        logging.debug(f"messages: {list(messages)}")
        response = cls.model.create(messages)

        revision = response.choices[0].message.content
        messages.append(cls._assistant_prompt.format(revision=revision))
        logger.info(revision)
        return messages

    @classmethod
    def interact(cls, messages, answer: str) -> tuple[list[dict], str]:
        messages.append(cls._user_prompt.format(answer=answer))
        logging.debug(f"messages: {list(messages)}")
        response = cls.model.create(messages)

        revision = response.choices[0].message.content
        logger.info(revision)
        messages.append(cls._assistant_prompt.format(revision=revision))
        return messages
