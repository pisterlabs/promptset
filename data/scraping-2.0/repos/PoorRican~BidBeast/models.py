from enum import IntEnum
from typing import Union, List, Optional, Iterator
from uuid import UUID, uuid4

from langchain.pydantic_v1 import BaseModel, Field
from postgrest.types import CountMethod

from db import SUPABASE


class Viability(IntEnum):
    NULL = -1
    DISLIKE = 0
    LIKE = 1

    @property
    def text(self) -> str:
        if self == self.LIKE:
            return "Would Bid"
        elif self == self.DISLIKE:
            return "Would Not Bid"
        else:
            return "Ambiguous"


class FeedbackModel(BaseModel):
    pros: List[str] = Field(description="appealing aspects of this job", default=[])
    cons: List[str] = Field(description="unappealing aspects of this job", default=[])
    viability: Viability = Field(description="final decision to bid or not", default=Viability.NULL)

    def upload(self, uuid: UUID):
        SUPABASE.table('potential_jobs') \
            .update({'viability': self.viability.value,
                     'pros': self.pros,
                     'cons': self.cons,
                     'reviewed': True
                     }) \
            .eq('id', uuid) \
            .execute()


class Job(object):
    """ Job object to store title and description """
    id: UUID
    title: str
    description: str
    link: str
    summary: str = ''
    feedback: Union[FeedbackModel, None] = None

    def __init__(self, title: str, description: str, link: str):
        self.id = uuid4()
        self.title = title
        self.description = description
        self.link = link

    def __repr__(self):
        return self.title

    @staticmethod
    def from_row(row: dict) -> 'Job':
        """ generates a fully populated `Job` object from a row """
        # job data
        for key in ('id', 'title', 'summary', 'desc', 'link'):
            assert key in row.keys()
        job = Job(row['title'], row['desc'], row['link'])
        job.id = row['id']
        job.summary = row['summary']

        try:
            for key in ('viability', 'pros', 'cons'):
                assert key in row.keys()
            fb = FeedbackModel(pros=row['pros'],
                               cons=row['cons'],
                               viability=row['viability'])
            job.feedback = fb
        except AssertionError:
            pass
        return job

    def summary_repr(self, index: Optional[int] = None) -> str:
        prefix = '##'
        if index is not None:
            prefix += f" {index+1}."
        return f"{prefix} {self.title}\n{self.summary}\n\n{self.link}"

    def detailed_repr(self) -> Iterator[str]:
        yield f"\n\n# {self.title}\n"

        # divide description into chunks of 2000 characters
        chunk_len = 2000
        for i in range(0, len(self.description), chunk_len):
            yield f"\n{self.description[i:i + chunk_len]}"

    @staticmethod
    def fetch_ambiguous(*_) -> list['Job']:
        jobs = []
        results = SUPABASE.table("potential_jobs") \
            .select("id", "title", "desc", count=CountMethod.exact) \
            .eq('viability', -1) \
            .execute()
        data = results.data
        if data:
            for row in data:
                job = Job(row['title'], row['desc'], '')
                job.id = row['id']
                jobs.append(job)
        return jobs

    @staticmethod
    def fetch_unreviewed(*_) -> list['Job']:
        results = SUPABASE.table('potential_jobs') \
            .select('id, title, summary, desc, link, viability, cons, pros',
                    count=CountMethod.exact) \
            .eq('reviewed', False) \
            .execute()
        jobs = []
        for data in results.data:
            job = Job.from_row(data)
            jobs.append(job)
        return jobs
