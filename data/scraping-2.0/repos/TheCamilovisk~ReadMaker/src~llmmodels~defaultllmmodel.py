from typing import Dict

from langchain.chat_models import ChatOpenAI
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser

from src.configs.llmmodels.defaultllmmodel import DefaultLLMModelConfig
from src.repositoryadapters.baseadapter import BaseRepositoryAdapter
from src.utils.preprocess import remove_markdown_tags
from src.utils.prompt import (
    execute_prompt,
    get_files_structure_text,
    get_files_summaries_text,
)

from .basellmmodel import BaseLLMModel


class DefaultLLMModel(BaseLLMModel):
    def __init__(
        self,
        repo: BaseRepositoryAdapter,
        config: DefaultLLMModelConfig = DefaultLLMModelConfig.get_default_config(),
    ):
        self.config = config
        self.repo = repo
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
        self.file_summary_chain = (
            self._get_file_summary_prompt() | self.llm | StrOutputParser()
        )
        self.introduction_chain = (
            self._get_introduction_prompt() | self.llm | StrOutputParser()
        )
        self.installation_chain = (
            self._get_installation_prompt() | self.llm | StrOutputParser()
        )
        self.repository_overview_chain = (
            self._get_repository_overview_prompt() | self.llm | StrOutputParser()
        )

    def _create_prompt(self, template_file_path: str) -> BaseChatPromptTemplate:
        prompt = ChatPromptTemplate.from_template(template_file_path)
        return prompt

    def _get_file_summary_prompt(self) -> BaseChatPromptTemplate:
        return self._create_prompt(self.config.file_summary_prompt_template)

    def _get_introduction_prompt(self) -> BaseChatPromptTemplate:
        return self._create_prompt(self.config.introduction_prompt_template)

    def _get_file_structure_from_repo(self) -> str:
        file_structure = self.repo.repo_structure(
            directories_only=self.config.directories_only_in_file_structure
        )
        output_text = f"# File Structure\n\n```\n{file_structure}\n```"
        return output_text

    def _get_installation_prompt(self) -> BaseChatPromptTemplate:
        return self._create_prompt(self.config.installation_prompt_template)

    def _get_repository_overview_prompt(self) -> BaseChatPromptTemplate:
        return self._create_prompt(self.config.repository_overview_prompt_template)

    def _get_license_from_repo(self) -> str:
        license_type, license_link = self.repo.license()
        output_text = f"# License\n\n[{license_type}]({license_link})"
        return output_text

    def _get_files_summaries(self, files_contents: Dict[str, str]) -> Dict[str, str]:
        files_summaries = {}
        for file, contents in files_contents.items():
            print(f"Summarizing file: {file}")
            summary = execute_prompt(self.file_summary_chain, file_contents=contents)
            files_summaries[file] = summary
        return files_summaries

    def generate_readme(self) -> str:
        files_structure_text = get_files_structure_text(self.repo.repo_list())

        files_summaries = self._get_files_summaries(self.repo.repo_files_contents())
        files_summaries_text = get_files_summaries_text(files_summaries)

        print("Generating README.md")
        readme_text = "\n\n".join(
            remove_markdown_tags(text)
            for text in (
                execute_prompt(
                    self.introduction_chain,
                    files_structure=files_structure_text,
                    files_summaries=files_summaries_text,
                ),
                self._get_file_structure_from_repo(),
                execute_prompt(
                    self.installation_chain,
                    files_summaries=files_summaries_text,
                    repository_url=self.repo.repo_url,
                ),
                execute_prompt(
                    self.repository_overview_chain,
                    files_structure=files_structure_text,
                    files_summaries=files_summaries_text,
                ),
                self._get_license_from_repo(),
            )
        )
        print("README.md generated")
        return readme_text
