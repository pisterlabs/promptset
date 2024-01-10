from datetime import datetime
import logging
from pprint import pprint
from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
from typing import Optional

from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, List
import jmespath as jm


@dataclass_json
@dataclass
class JobInfo:
    job_posting_id: str
    title: str
    company_name: str
    location: str
    description: str
    listed_at: int
    summary: Optional["SummaryJob"] = None
    linkedin_source_job: Optional[dict] = None

    EXPRESSION = {
        "job_posting_id": "jobPostingId",
        "listed_at": "listedAt",
        "title": "title",
        "company_name": 'companyDetails."com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany".companyResolutionResult.name',
        "location": "formattedLocation",
        "description": "description.text"
    }

    @staticmethod
    def from_job_info(job_info) -> "JobInfo":
        data = {k: jm.search(exp, job_info) for k, exp in JobInfo.EXPRESSION.items()}

        return JobInfo(
            # job_posting_id=job_info["jobPostingId"],
            # listed_at=job_info["listedAt"],
            # title=job_info["title"],
            # company_name=job_info["companyDetails"][
            #     "com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany"
            # ]["companyResolutionResult"]["name"],
            # location=job_info["formattedLocation"],
            # description=job_info["description"]["text"],
            **data,
            linkedin_source_job = job_info,
        )

@dataclass_json
@dataclass
class LinkedinSearchLoad:
    loadtime: datetime
    keywords: str
    listed_offset_seconds: int
    jobs_reply: list 

    def get_path_part(self):
        return self.loadtime.strftime("%Y/%m/%d")

    def get_file_name(self, extension=".json"):
        return f"{self.loadtime.strftime('%Y-%m-%d')}_{self.keywords.replace(' ', '_')}{extension}"


@dataclass_json
@dataclass
class SummaryJob:
    """Summary of a job description from LinkedIn."""

    location: str
    remote_from_spain: bool
    salary: str
    role_responsabilities: List[str]
    required_experience: List[str]
    benefits: List[str]


class SummaryJobLangChain(BaseModel):
    """Summary of a job description from LinkedIn."""

    location: str = Field(..., description="the location where the company is based from")
    remote_from_spain: bool = Field(..., description="whether it allows to work from Spain remotely")
    salary: str = Field(..., description="the salary that is going to be paid")
    role_responsabilities: List[str] = Field(..., description="the list of responsabilities and technologies for the role")
    required_experience: List[str] = Field(..., description="what experience the job requires")
    benefits: List[str] = Field(..., description="any extra benefits that come with the job")

    def to_dataclass(self):
        return SummaryJob.from_dict(self.dict())