"""Utility functions for accessing a repository."""
import os
from pathlib import Path
from typing import Optional

import magic
from langchain import tools as lc_tools
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.tools.file_management.utils import BaseFileToolMixin
from pathspec import pathspec


def list_files_in_repo(repo_path: str | Path, additional_gitignore_content: str = "") -> list[Path]:
    """List all files in a repository, excluding files and directories that are ignored by git."""
    gitignore_content = f".*\n{additional_gitignore_content}\n{_read_gitignore(repo_path)}"
    spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore_content.splitlines())

    file_list = []
    for root, dirs, files in os.walk(repo_path):
        root = Path(root)
        # Remove excluded directories from the list to prevent os.walk from processing them
        dirs[:] = [d for d in dirs if not spec.match_file((root / d).relative_to(repo_path))]

        for file in files:
            file_path = root / file
            rel_file_path = file_path.relative_to(repo_path)
            if not spec.match_file(rel_file_path) and _is_text_file(file_path):
                file_list.append(rel_file_path)

    file_list.sort(key=lambda p: (p.as_posix().lower(), p.as_posix()))
    return file_list


def _read_gitignore(repo_path: str | Path) -> str:
    gitignore_path = Path(repo_path) / ".gitignore"
    if not gitignore_path.is_file():
        return ""

    with open(gitignore_path, "r", encoding="utf-8") as file:
        gitignore_content = file.read()
    return gitignore_content


def _is_text_file(file_path: str | Path):
    file_mime = magic.from_file(file_path, mime=True)
    # TODO is this the exhaustive list of mime types that we want to index ?
    return file_mime.startswith("text/") or file_mime.startswith("application/json")


class ListRepoTool(BaseFileToolMixin, lc_tools.BaseTool):
    """Tool that lists all the files in a repo."""

    name: str = "list_repo"
    description: str = "List all the files in `%s` repo"
    repo_name: str = None

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if not self.repo_name:
            self.repo_name = Path(self.root_dir).name
        self.description = self.description % self.repo_name

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        file_list: list[Path] = list_files_in_repo(self.root_dir)

        file_list_str = "\n".join([file.as_posix() for file in file_list])
        result = f"Here is the complete list of files that can be found in `{self.repo_name}` repo:\n{file_list_str}"
        return result

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run()


class ReadFileTool(lc_tools.ReadFileTool):
    async def _arun(
        self,
        file_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self.run(file_path, run_manager=run_manager)


class WriteFileTool(lc_tools.WriteFileTool):
    async def _arun(
        self,
        file_path: str,
        text: str,
        append: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self.run(file_path, text=text, append=append, run_manager=run_manager)
