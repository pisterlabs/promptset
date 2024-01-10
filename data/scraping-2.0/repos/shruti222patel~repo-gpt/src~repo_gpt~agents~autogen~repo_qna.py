import logging
import os
import re
from pathlib import Path

import autogen
import pytest

from repo_gpt.agents.autogen.user_proxy_function_call_agent import (
    UserProxyFunctionCallAgent,
)
from repo_gpt.agents.repo_comprehender import get_relative_path_directory_structure
from repo_gpt.code_manager.abstract_extractor import AbstractCodeExtractor
from repo_gpt.file_handler.generic_code_file_handler import PythonFileHandler
from repo_gpt.openai_service import OpenAIService
from repo_gpt.search_service import SearchService, convert_search_df_to_json

logger = logging.getLogger(__name__)
from repo_gpt.logging_config import VERBOSE_INFO, configure_logging


class RepoQnA:
    def __init__(self, question, root_path, embedding_path=None, openai_api_key=None):
        self.question = question
        self.question_answer = None
        self.is_valid_answer = None
        self.validator_explanation = None
        self.openai_api_key = openai_api_key
        self.root_path = Path(root_path)
        self.embedding_path = (
            Path(embedding_path)
            if embedding_path is not None
            else self.root_path / ".repo_gpt/code_embeddings.pkl"
        )
        self.openai_service = OpenAIService(openai_api_key)
        self.search_service = SearchService(self.openai_service, self.embedding_path)
        self.openai_api_key = (
            openai_api_key if openai_api_key else os.environ["OPENAI_API_KEY"]
        )

        config_list = [
            {
                "model": "gpt-3.5-turbo-1106",
                "api_key": self.openai_api_key,
            },  # OpenAI API endpoint for gpt-3.5-turbo
        ]
        self.config = self.create_function_augmented_config(config_list)

    def create_code_librarian(self):
        return autogen.AssistantAgent(
            name="CodeRepo Librarian",
            system_message="""You are a detail-oriented world-class software engineer. You specialize in answering questions about the user's codebase. You use the functions to search and understand the codebase.""",
            llm_config=self.config,
        )

    def is_answer_to_question(self, msg):
        is_termination_msg = msg.get("content", "") and msg.get(
            "content", ""
        ).rstrip().startswith("ANSWER:")

        if is_termination_msg:
            self.question_answer = msg.get("content", "").replace("ANSWER:", "").strip()

        return is_termination_msg

    def is_answer_to_question(self, msg):
        content = msg.get("content", "")
        if not content:
            return False

        # Check if the content contains the "ANSWER:" marker
        if "ANSWER:" in content:
            # Find the position where "ANSWER:" ends
            answer_start_index = content.find("ANSWER:") + len("ANSWER:")

            # Extract the content after "ANSWER:"
            self.question_answer = content[answer_start_index:].strip()

            # Return True since the content contains an answer
            return True

        # Return False if the content does not contain "ANSWER:"
        return False

    def is_answer_correct(self, msg):
        content = msg.get("content", "")
        if not content:
            return False

        # Check for "ANSWER:" and capture the value after it
        answer_pattern = r"^ANSWER:\s*(True|False|Unsure)"
        answer_match = re.search(answer_pattern, content, re.IGNORECASE)

        # If "ANSWER:" is found, extract the value and check for "EXPLANATION:"
        if answer_match:
            self.is_valid_answer = answer_match.group(1)

            # Check for "EXPLANATION:" and capture the value after it
            explanation_pattern = r"EXPLANATION:\s*(.*)"
            explanation_match = re.search(explanation_pattern, content)
            if explanation_match:
                self.validator_explanation = explanation_match.group(1).strip()

            return True
        return False

    def create_code_answer_validator(self):
        return autogen.AssistantAgent(
            name="CodeRepo Answer Validator",
            system_message="""You are a detail-oriented world-class software engineer. You specialize in criticing answers to code repository questions. You do this by searching and understanding the codebase to check if the provided answer is correct. You use the functions to search and understand the codebase.""",
            llm_config=self.config,
        )

    def create_function_augmented_config(self, config):
        return {
            "functions": [
                {
                    "name": "semantic_search",
                    "description": "Use this function to search the entire codebase semantically. The input should be the search query string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": f"""The semantic search query to use to search the code base.""",
                            }
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "view_function_code",
                    "description": "Use this function to search for and view a function's code in the user's codebase. Input should be the name of the function you want to search for.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "function_name": {
                                "type": "string",
                                "description": f"""The name of the function or its description.""",
                            }
                        },
                        "required": ["function_name"],
                    },
                },
                #                 {
                #                     "name": "view_file_functions_and_classes",
                #                     "description": "Use this function to retrieve a list of the functions and classes in a file from the user's codebase.",
                #                     "parameters": {
                #                         "type": "object",
                #                         "properties": {
                #                             "file_paths": {
                #                                 "type": "array",
                #                                 "items": {
                #                                     "type": "string",
                #                                     "description": "An array of one or more file paths of a file you want to retrieve functions and classes from. If a file doesn't exist, the function will return a string saying so.",
                #                                 },
                #                                 "description": f"""The file paths of the files you want to retrieve functions and classes to understand the user's task better. Below are the files within the user's repository:
                # {get_relative_path_directory_structure("/Users/shrutipatel/projects/work/repo-gpt")}
                # """,
                #                             }
                #                         },
                #                         "required": ["file_paths"],
                #                     },
                #                 },
                {
                    "name": "view_raw_file_contents",
                    "description": "Use this function to retrieve the raw contents of a file from the user's codebase.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": f"""The file path of the file you want to retrieve the raw contents of. Below are the files within the user's repository:
{get_relative_path_directory_structure("/Users/shrutipatel/projects/work/repo-gpt")}
""",
                            }
                        },
                        "required": ["file_path"],
                    },
                },
            ],
            "config_list": config,
            "request_timeout": 120,
        }

    def create_user_proxy(self, is_termination_msg):
        return self.UserProxyAgent(
            self.root_path,
            self.embedding_path,
            self.openai_api_key,
            is_termination_msg,
        )

    def initiate_chat(self, validate_answer=False):
        code_librarian = self.create_code_librarian()
        user_proxy = self.create_user_proxy(self.is_answer_to_question)
        user_proxy.initiate_chat(
            code_librarian,
            message=f"""{self.question}
Use the functions to read and understand the codebase. Then, once you have an answer, backed up by searching in the code, reply with:
```
ANSWER:<answer>
```
If you cannot answer the user's question, reply with:
```
ANSWER: I cannot answer this question.
```
""",
            clear_history=True,
        )

        # print(self.question_answer)
        if self.question_answer is not None and validate_answer:
            # Validate answer
            code_answer_validator = self.create_code_answer_validator()
            validator_user_proxy = self.create_user_proxy(self.is_answer_correct)

            validator_user_proxy.initiate_chat(
                code_answer_validator,
                message=f"""Verify the following question & answer are correct:
Q:{self.question}
A:{self.question_answer}
------------------
Use the functions to read and understand the codebase. Then, once you have an answer, backed up by searching in the code, reply with:
```
ANSWER:<True or False or Unsure>
EXPLANATION: <your explanation>
```
""",
                clear_history=True,
            )

            # print(self.is_valid_answer)

            # TODO add reflection

            if not self.is_valid_answer:
                user_proxy.initiate_chat(
                    code_librarian,
                    message=f"""{self.question}
This is the last answer, incorrect answer:
{self.question_answer}
Here is an explanation of why the answer is incorrect:
{self.validator_explanation}
--------
Use the functions to read and understand the codebase. Then, once you have an answer, backed up by searching in the code, reply with:
```
ANSWER:<answer>
```
If you cannot answer the user's question, reply with:
```
ANSWER: I cannot answer this question.
```
""",
                    clear_history=True,
                )
        return self.question_answer

    class UserProxyAgent(UserProxyFunctionCallAgent):
        def __init__(
            self, root_path, embedding_path, openai_api_key, is_termination_msg_func
        ):
            self.openai_service = OpenAIService(openai_api_key)
            self.root_path = root_path
            self.embedding_path = embedding_path
            self.search_service = SearchService(
                self.openai_service, self.embedding_path
            )

            super().__init__(
                name="You",
                is_termination_msg=is_termination_msg_func,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=3,
                code_execution_config=False,
                function_map={
                    # "view_file_functions_and_classes": self.view_file_functions_and_classes,
                    "view_raw_file_contents": self.view_raw_file_contents,
                    "view_function_code": self.view_function_code,
                    "semantic_search": self.semantic_search,
                },
            )

        def view_function_code(self, function_name):
            # logger.info(f"Reading the code for: {function_name}")
            df = self.search_service.find_function_match(function_name)

            if df is None or df.empty:
                return "Not found."
            else:
                return convert_search_df_to_json(df, ["code"])

        def semantic_search(self, query):
            # logger.info(f"Searching the codebase for: {query}")
            return convert_search_df_to_json(
                self.search_service.semantic_search_similar_code(query)
            )

        # def view_file_functions_and_classes(self, file_paths):
        #     # logger.info(f"Skimming the code in: {file_paths}")
        #     results = []
        #     for file_path in file_paths:
        #         full_path = self.root_path / Path(file_path)
        #
        #         if not full_path.exists():
        #             results.append(f"File not found: {file_path}")
        #             continue  # Skip to the next iteration
        #         elif full_path.is_dir():
        #             results.append(
        #                 f"This is not a file, but a directory, pass a filepath instead: {file_path}"
        #             )
        #             continue  # Skip to the next iteration
        #
        #         parsable_extensions = AbstractCodeExtractor.get_file_extensions_with_handlers()
        #         if full_path.suffix not in parsable_extensions:
        #             return f"Cannot parse file." # TODO just pass the full text instead
        #         file_handler = AbstractCodeExtractor.get_handler(file_path)()
        #         file_contents = file_handler.summarize_file(full_path)
        #
        #         if "" == file_contents:
        #             results.append(f"File is empty: {file_path}")
        #             continue
        #
        #         results.append(file_contents)
        #
        #     return "\n".join(results)

        def view_raw_file_contents(self, file_path):
            partial_path = Path(file_path)
            full_path = self.root_path / Path(file_path)
            print(full_path)

            path_to_read = None

            if not full_path.exists() and not partial_path.exists():
                return f"File not found: {file_path}"
            elif full_path.exists() and not full_path.is_dir():
                path_to_read = full_path
            elif partial_path.exists() and not partial_path.is_dir():
                path_to_read = partial_path
            else:
                return f"This is not a file, but a directory, pass a filepath instead: {file_path}"

            with open(path_to_read, "r") as f:
                return f.read()


def test():
    # REPO_QUESTION = "How do I add a handler for a new language to the codebase?"
    # REPO_QUESTION = "Where should I add new tests for a new Java file handler I'm writing?"
    # REPO_QUESTION = "What is the framework used for writing tests?"
    REPO_QUESTION = "How do I replace pandas with polars code?"
    root_path = "/Users/shrutipatel/projects/work/repo-gpt/"
    repo_qna = RepoQnA(REPO_QUESTION, root_path)
    print(repo_qna.initiate_chat())
    assert False
