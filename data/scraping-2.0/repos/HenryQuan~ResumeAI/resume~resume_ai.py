"""
Wrapper for the OpenAI API calls
"""

import openai
import os
import builtins
import re
from enum import Enum

DEBUG = True


# override print function to print only in debug mode
def print(*args, **kwargs):
    if DEBUG:
        builtins.print(*args, **kwargs)


class ResumeInfo:
    """
    Information required for the AI to score, review or rewrite the resume.
    additional - this is appended at the end
    instruction - this is prepended at the start
    """

    def __init__(
        self, resume: str, job_post: str, additional: str = "", instruction: str = ""
    ) -> None:
        self.resume = resume
        self.job_post = job_post
        self.additional = additional
        self.instruction = instruction

    def generate_prompt(self, role_description: str) -> str:
        """
        Generate the prompt for the AI
        """
        return (
            f"Instruction: {self.instruction}\n###\n{role_description}\n###\n"
            + f"Resume:\n{self.resume}\n###\n"
            + f"Job post:\n{self.job_post}\n###\n{self.additional}"
        )


class ResumeAIRole(Enum):
    HR = 0
    APPLICANT = 1

    def description(self):
        # Modify the role can result in significant changes in the output
        if self == ResumeAIRole.HR:
            return "You are the principal HR (Human Resources)/recruiter of a company reviewing candidates for a new position. Any information which is not relevant to the job post should be ignored. Anything which is not mentioned in the resume should be considered as false."
        elif self == ResumeAIRole.APPLICANT:
            return "You are a candidate applying for a new job position and working on the resume. Refrain from adding any fake/not given information in the resume. Avoid stating any information which it not true or relevant to the job post."
        raise ValueError("Invalid role")


class ResumeAI:
    """
    OpenAI with custom instructions
    """

    def __init__(self, role: ResumeAIRole) -> None:
        with open(f"{os.path.dirname(__file__)}/openai.key") as f:
            key = f.read().strip()
        openai.api_key = key
        print("OpenAI API key loaded with role:", role)
        self._role: ResumeAIRole = role

    def ask(self, message: str) -> str:
        """
        Ask a question to the AI
        """
        role_description = self._role.description()

        final_message = f"{role_description}\n{message}"
        response = self._ask_turbo(final_message)
        output = response["choices"][0]["message"]["content"]
        # print token used
        print("OpenAI token used:", response["usage"]["total_tokens"])
        return output.strip()

    def score_and_review(self, info: ResumeInfo) -> str:
        """
        Rate the resume based on the job post depending on closely the resume matches the job post
        """
        info.additional = (
            "Review the resume from your perspective. Give feedback, suggestions or improvements. "
            + "Indicate if the resume represents a good match with the job post."
            + "Output the score (0 - 10), review (one summary) and fit (true/false) in the format: Score: <score>\nReview: <review>\nFit: <fit> without any other text."
        )
        info.instruction = (
            "The review should be based on how closely the job post and the resume match."
            + "Note that there is either a good match or not. There is no in-between."
        )
        result = self.custom(info)
        return result

    def score_and_review_avg(self, info: ResumeInfo, repeat: int = 3) -> str:
        """
        Average the score and review with multiple runs to hopefully get a more accurate result
        """
        results = []
        for _ in range(repeat):
            result = self.score_and_review(info)
            print(result)
            results.append(result)
        # use it as the input to get the final result
        final_message = (
            "\n".join(results)
            + "\n###\nWith the above results, output the final score (0 - 10), review (one summary) and fit (true/false) in the format: Score: <score>\nReview: <review>\nFit: <fit> without any other text."
        )
        final_result = self._retrieve_output(self._ask_turbo(final_message))
        return final_result

    def rewrite(self, info: ResumeInfo) -> str:
        """
        Rewrite the resume based on the job post
        """
        info.additional = (
            "Rewrite the resume and cherrypick things matching the job post to get as close as the posting.\nGenerate the new resume in a clear format.\nApply proper spacing and indentation.\n"
            + "Add a cover letter highlighting why this resume is a good match for the job post.\n"
            + "Output in the MarkDown format without any other text starting with # Resume"
        )

        return self.custom_output(info, "resume/new_resume.md")

    def custom(self, info: ResumeInfo) -> str:
        """
        Use this function for any custom instructions
        """
        role_description = self._role.description()
        final_message = info.generate_prompt(role_description)
        response = self._ask_turbo(final_message)
        return self._retrieve_output(response)

    def custom_output(self, info: ResumeInfo, file_name: str) -> str:
        """
        Use this function for any custom instructions with output to a file
        """
        output = self.custom(info)
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)
        return output

    def _ask_turbo(self, message: str) -> dict:
        """
        Ask a question to the AI with the turbo engine
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "assistant",
                    "content": message,
                },
            ],
        )
        return response

    def _retrieve_output(self, response: dict) -> str:
        """
        Retrieve the output from the response
        """
        return response["choices"][0]["message"]["content"].strip()

    def shrink_input(self, message: str) -> str:
        """
        Remove extra spaces and newlines
        """
        return re.sub(r"\s+", " ", message).strip()


if __name__ == "__main__":
    from resume_extract import extract_resume

    resume = ResumeAI(ResumeAIRole.HR)
    # print(resume.ask("おなまえは？"))
    # exit()

    # review
    # my_resume = extract_resume("resume/resume.pdf")
    with open("resume/resume.txt", encoding="utf-8") as f:
        my_resume = f.read()
    my_resume = resume.shrink_input(my_resume)
    with open("resume/job_post.txt", encoding="utf-8") as f:
        job_post = f.read()
    job_post = resume.shrink_input(job_post)
    resume_info = ResumeInfo(my_resume, job_post)
    # print(resume.score_and_review_avg(resume_info, 5))

    # rewrite
    print(resume.rewrite(resume_info))
