import validators
from dataclasses import dataclass
from typing import Optional
from serde.toml import from_toml, to_toml
from openai import OpenAI
import urllib.request


@dataclass
class ContactInfo:
    first_name: str
    last_name: str
    phone: str
    email: str
    linkedin: Optional[str]
    github: Optional[str]
    website: Optional[str]


@dataclass
class Degree:
    degree: str
    institution: str
    year: Optional[int]
    gpa: Optional[float]


@dataclass
class Job:
    employer: str
    title: str
    start_date: str
    end_date: Optional[str]
    description: Optional[list[str]]


@dataclass
class Resume:
    contact_info: ContactInfo
    skills: Optional[dict[str, list[str]]]
    education: Optional[list[Degree]]
    narrative: Optional[str]
    job_history: Optional[list[Job]]

    def write(self, path: str = "resume.toml") -> None:
        with open(path, "w") as f:
            f.write(to_toml(self))

    def convert(self) -> Optional[str]:
        to_toml(self)


class Curator:
    def __init__(
        self,
        resume: Resume,
        client: OpenAI,
        job_posting: str,
        model: str = "gpt-4-1106-preview",
    ):
        self.resume = resume
        self.client = client
        self.model = model
        self.job_posting = job_posting
        if validators.url(job_posting):
            uf = urllib.request.urlopen(job_posting)
            html = uf.read().decode("utf-8")
            chat = self.client.chat.completions.create(
                messages = [{
                    "role": "user",
                    "content": f"Please extract just the job information from the following following job posting HTML: ```{html}```"
                }],
                model=self.model
            )
            self.job_posting = chat.choices[0].message.content
            print("extracted job posting:")
            print(self.job_posting)

    def curate_skills(self) -> Optional[dict[str, list[str]]]:
        print("checking skills...")
        curated_skills = {}
        if self.resume.skills:
            for category, skills in self.resume.skills.items():
                completion = self.client.chat.completions.create(
                    messages = [{
                        "role": "user",
                        "content": f"Given the following job posting ```\n{self.job_posting}\n```, please select the most relevant skills related to the job posting and return as a list separated by the character '|' sorted by relevancy.  Please do not return any other skills other than those listed below.  If there is not enough relevant information, please return nothing:\n```{skills}\n```\n",
                    }],
                    model=self.model,
                )
                print(completion.choices[0].message.content, "\n**********\n")
                curated_skills[category] = completion.choices[0].message.content.split("|")
        if len(curated_skills) > 0:
            return curated_skills
        return None

    def curate_jobs(self) -> Optional[list[Job]]:
        print("checking jobs...")
        curated_jobs = []
        if self.resume.job_history:
            for job in self.resume.job_history:
                completion = self.client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": f"Given the job posting ```\n{self.job_posting}\n```, please select the most relevant experience from the following list related to the job posting and return as a list separated by the character '|' sorted by relevancy.  Please do not introduce any additional information other than that which is in the below summary.  If there is not enough relevant information, please return nothing:\n```{job.description}\n```\n",
                    }],
                    model=self.model,
                )
                print(completion.choices[0].message.content, "\n**********\n")
                curated_jobs.append(
                    Job(
                        job.employer,
                        job.title,
                        job.start_date,
                        job.end_date,
                        completion.choices[0].message.content.split("|"),
                    )
                )
        if len(curated_jobs) > 0:
            return curated_jobs
        return None

    def curate_narrative(self) -> Optional[str]:
        print("checking narrative...")
        if self.resume.narrative:
            completion = self.client.chat.completions.create(
                messages = [{
                    "role": "user",
                    "content": f"Given the job posting ```\n{self.job_posting}\n```, please rewrite the following narrative about yourself to make it the most relevant to the job posting.  Please do not use any details not explicitly discussed below.  If there are no relevant details, please return nothing:\n```\n{self.resume.narrative}\n```\n",
                }],
                model=self.model,
            )
            print(completion.choices[0].message.content, "\n**********\n")
            if completion.choices[0].message.content.strip() != "":
                return completion.choices[0].message.content
        return None

    def curate(self) -> Resume:
        curated_skills = self.curate_skills()
        curated_jobs = self.curate_jobs()
        curated_narrative = self.curate_narrative()
        return Resume(
            self.resume.contact_info,
            curated_skills,
            self.resume.education,
            curated_narrative,
            curated_jobs,
        )


def read_resume(path: str = "resume.toml") -> Resume:
    with open(path, "r") as f:
        return from_toml(Resume, f.read())
