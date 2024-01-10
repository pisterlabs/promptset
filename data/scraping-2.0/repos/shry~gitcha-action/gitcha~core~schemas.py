"""
Basic gitcha config
"""
import dataclasses
from pathlib import Path
from typing import Optional

from github.GitRelease import GitRelease
from langchain.docstore.document import Document
from pydantic import BaseModel, EmailStr, Field, HttpUrl, PastDate


class Address(BaseModel):
    """
    A given address
    """
    street_address: Optional[str]
    postal_code: Optional[str]
    city: Optional[str]
    region: Optional[str]
    country: Optional[str]


class Person(BaseModel):
    """
    All values of a person. 
    This is also the base of the gitcha config.

    """
    given_name: str
    family_name: Optional[str]
    pronouns: Optional[str]

    knows_language: Optional[list[str]]
    knows_coding: Optional[list[str]]  # Coding language

    nationality: Optional[str]
    phone: Optional[str]
    email: Optional[EmailStr]
    birth_date: Optional[PastDate]

    desired_salary: Optional[int]
    highest_lvl_education: Optional[str]

    address: Optional[Address]

    websites: Optional[list[HttpUrl]]


class Config(BaseModel):
    """
    The internal gitcha developer config
    """
    root_folder: Path = Field(default='/',
                              description='The root folder is the absolute base of all relative provided folder references'
                              )
    public_folder: Path = Field(default='/public',
                                description='All files of the public folder will be transfered along a job application.'
                                )
    work_history_folder: Path = Field(default='/work_history',
                                      description="""Provide for every job a seperate .md file in this folder."""
                                      )
    certificats_folder: str = Field(default='/certs')

    projects_folder: str = Field(default='/projects')

    job_posting_folder: str = Field(
        default='/job_postings', description='Paste your interesting job postings as .md files here')

    output_lang: str = Field(
        default='English', description='The default language of the AI generated content')


class GitchaYaml(Person):
    """
    The config file .gitcha.yml which every git project needs to have.
    """
    config: Optional[Config] = Config()


@dataclasses.dataclass
class ParsedDocs:
    """
    Information about the documents we analyzed
    """
    cv_files: list[Document] = dataclasses.field(default_factory=list)
    job_postings: list[Document] = dataclasses.field(default_factory=list)
    cv_summary: Optional[str] = None

    def total_files(self) -> int:
        """Return the number of files
        """
        return len(self.cv_files) + len(self.job_postings)


@dataclasses.dataclass
class RepoConfig:
    """
    Repo config
    """
    path: str
    name: Optional[str] = ''
    api_token: Optional[str] = ''
    ref: Optional[str] = ''
    release: Optional[GitRelease] = None
    gitcha: Optional[GitchaYaml] = None
