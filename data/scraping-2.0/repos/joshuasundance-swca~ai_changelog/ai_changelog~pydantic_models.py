"""Pydantic models for ai_changelog"""

from __future__ import annotations

import os
import subprocess
from typing import List, Optional

from langchain.pydantic_v1 import BaseModel, Field

from ai_changelog.string_templates import markdown_template


class Commit(BaseModel):
    """A commit"""

    commit_hash: str = Field(..., description="The commit hash")
    date_time_str: str = Field(..., description="Formatted date and time str")
    diff: str = Field(..., description="The diff for the commit")


class CommitDescription(BaseModel):
    """A commit description"""

    short_description: str = Field(
        ...,
        description="A technical and concise description of changes implemented in the commit",
    )
    long_description: List[str] = Field(
        ...,
        description="Markdown bullet-point formatted list of changes implemented in the commit",
    )


class CommitInfo(Commit, CommitDescription):
    """A commit and its description"""

    @staticmethod
    def get_repo_name(repo_path: Optional[str] = None) -> str:
        """Get the repo name from the remote origin URL"""
        repo_path = repo_path or os.getcwd()
        os.chdir(repo_path)
        return (
            subprocess.check_output(["git", "remote", "get-url", "origin"])
            .decode()
            .replace("https://github.com/", "")
            .replace(".git", "")
            .strip()
        )

    def markdown(self) -> str:
        """Generate markdown for the commit info"""
        _repo_name = self.get_repo_name()
        bullet_points = "\n".join(
            [f"- {line.strip('*- ').strip()}" for line in self.long_description],
        ).strip()
        return markdown_template.format(
            short_description=self.short_description,
            commit_hash=self.commit_hash,
            bullet_points=bullet_points,
            repo_name=_repo_name,
            date_time_str=self.date_time_str,
        )

    @staticmethod
    def infos_to_str(infos: List[CommitInfo]) -> str:
        """Convert a list of CommitInfo objects to a string"""
        return "\n".join([info.markdown().strip() for info in infos]).strip()
