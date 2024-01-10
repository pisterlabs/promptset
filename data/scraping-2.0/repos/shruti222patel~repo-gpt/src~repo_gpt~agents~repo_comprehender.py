# Refactored RepoUnderstandingAgent using the ParentAgent
import logging
import os
from pathlib import Path

from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from tqdm import tqdm

from repo_gpt.agents.base_agent import BaseAgent
from repo_gpt.file_handler.generic_code_file_handler import PythonFileHandler
from repo_gpt.openai_service import OpenAIService
from repo_gpt.search_service import SearchService, convert_search_df_to_json

# Initialize the tqdm integration with pandas


logger = logging.getLogger(__name__)


def get_gitignore_spec(root_directory):
    gitignore_file = os.path.join(root_directory, ".gitignore")
    if not os.path.exists(gitignore_file):
        return None
    with open(gitignore_file, "r") as f:
        spec = PathSpec.from_lines(GitWildMatchPattern, f)
    return spec


def is_hidden(path):
    # Check if a file or directory is hidden by checking if its name starts with a dot
    return os.path.basename(path).startswith(".")


def get_indented_directory_structure(root_directory):
    structured_output = []
    gitignore_spec = get_gitignore_spec(root_directory)

    for current_path, directories, files in os.walk(root_directory):
        # Filter out hidden directories and those in gitignore
        directories[:] = [
            d
            for d in directories
            if not is_hidden(d)
            and (
                not gitignore_spec
                or not gitignore_spec.match_file(os.path.join(current_path, d))
            )
        ]

        # Skip hidden directories in the main loop
        if is_hidden(current_path):
            continue

        depth = current_path.replace(root_directory, "").count(os.sep)
        indent = "    " * depth
        structured_output.append(f"{indent}/{os.path.basename(current_path)}")
        sub_indent = "    " * (depth + 1)

        for file in sorted(files):
            # Skip hidden files or those in gitignore
            if not is_hidden(file) and (
                not gitignore_spec
                or not gitignore_spec.match_file(os.path.join(current_path, file))
            ):
                structured_output.append(f"{sub_indent}{file}")

    return "\n".join(structured_output)


def get_relative_path_directory_structure(root_directory):
    structured_output = []
    gitignore_spec = get_gitignore_spec(root_directory)

    for current_path, directories, files in os.walk(root_directory):
        # Filter out hidden directories and those in gitignore
        directories[:] = [
            d
            for d in directories
            if not is_hidden(d)
            and (
                not gitignore_spec
                or not gitignore_spec.match_file(os.path.join(current_path, d))
            )
        ]

        # Skip hidden directories in the main loop
        if is_hidden(current_path):
            continue

        # # Convert the current directory path to a relative path from the root directory
        rel_dir = os.path.relpath(current_path, root_directory)

        # # Append the relative directory path to structured_output
        # structured_output.append(rel_dir if rel_dir != "." else "")

        for file in sorted(files):
            # Skip hidden files or those in gitignore
            if not is_hidden(file) and (
                not gitignore_spec
                or not gitignore_spec.match_file(os.path.join(current_path, file))
            ):
                # Combine the relative directory path with the file name to get the relative file path
                rel_file_path = os.path.join(rel_dir, file)
                structured_output.append(rel_file_path)

    return structured_output


def get_relative_path_directory_structure_string(root_directory):
    return "\n".join(get_relative_path_directory_structure(root_directory))


# print(get_relative_path_directory_structure_string("/Users/shrutipatel/projects/work/repo-gpt"))


class RepoUnderstandingAgent(BaseAgent):
    system_prompt = """You are an expert software engineer on a specific code repository. Users ask you how they can implement something in their codebase. You first use your tools to search and understand the codebase and then figure out how to implement the users' task in the repository.
    **DO NOT** communicate with the user directly. Use the functions instead.
    """

    def __init__(
        self,
        user_task,
        root_path,
        system_prompt=system_prompt,
        threshold=10,
        debug=False,
        openai_key=None,
    ):
        self.system_prompt = system_prompt
        super().__init__(
            user_task,
            "create_plan_to_complete_user_task",
            system_prompt,
            threshold,
            debug,
        )  # Call ParentAgent constructor
        self.root_path = root_path
        self.embedding_path = self.root_path / ".repo_gpt/code_embeddings.pkl"
        self.openai_service = (
            OpenAIService() if not openai_key else OpenAIService(openai_key)
        )
        self.search_service = SearchService(self.openai_service, self.embedding_path)
        self.pythonfilehandler = (
            PythonFileHandler()
        )  # TODO: update to handle more than python files (all except SQL)

        self.functions = self._initialize_functions()

    def _initialize_functions(self):
        # Define function details
        return [
            {
                "name": "semantic_search",
                "description": "Use this function to search the entire codebase semantically. The input should be the search query string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                            The semantic search query to use to search the code base.
                            """,
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "view_function_code",
                "description": "Use this function to search for and view a function's code in the user's codebase. Input should be the name of the function you want to search for. An empty response means the given files don't exist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": f"""
                            The name of the function or its description.
                            """,
                        }
                    },
                    "required": ["function_name"],
                },
            },
            {
                "name": "view_file_functions_and_classes",
                "description": "Use this function to retrieve a list of the functions and classes in a file from the user's codebase. An empty response means the given files don't exist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_paths": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "An array of one or more file paths of a file you want to retrieve functions and classes from. If a file doesn't exist, the function will return a string saying so.",
                            },
                            "description": f"""
                        The file paths of the files you want to retrieve functions and classes for to better understand the user's task. Below are the files within the user's repository:
                        {get_relative_path_directory_structure("/Users/shrutipatel/projects/work/repo-gpt")}
                        """,
                        }
                    },
                    "required": ["file_paths"],
                },
            },
            {
                "name": "create_plan_to_complete_user_task",
                "description": "Use this function when you understand the user's task and have a detailed plan ready for completing the user's task. The input should be a step-by-step plan on how to complete the user's task. It can include things like 'Create a new file with a given file path', 'Add the given code to the file', etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "string",
                            "description": f"""
                            A step-by-step plan on how to complete the user's task. It can include things like "Create a new file with a given file path", "Add the given code to the file", etc.
                            """,
                        }
                    },
                    "required": ["plan"],
                },
            },
        ]

    def view_function_code(self, function_name):
        logger.info(f"Reading the code for: {function_name}")
        functions_df, classes_df = self.search_service.find_function_match(
            function_name
        )

        if (classes_df is None or classes_df.empty) and (
            functions_df is None or functions_df.empty
        ):
            return ""
        elif functions_df is None or functions_df.empty:
            return convert_search_df_to_json(classes_df)
        elif classes_df is None or classes_df.empty:
            return convert_search_df_to_json(functions_df)
        else:
            return convert_search_df_to_json(functions_df)

    def semantic_search(self, query):
        logger.info(f"Searching the codebase for: {query}")
        return convert_search_df_to_json(
            self.search_service.semantic_search_similar_code(query)
        )

    def view_file_functions_and_classes(self, file_paths):
        logger.info(f"Skimming the code in: {file_paths}")
        results = []
        for file_path in file_paths:
            full_path = self.root_path / Path(file_path)

            if not full_path.exists():
                results.append(f"File not found: {file_path}")
                continue  # Skip to the next iteration
            elif full_path.is_dir():
                results.append(
                    f"This is not a file, but a directory, pass a filepath instead: {file_path}"
                )
                continue  # Skip to the next iteration

            # TODO select the correct filehandler and then summarize file
            results.append(self.pythonfilehandler.summarize_file(full_path))

        return "\n".join(results)

    def create_plan_to_complete_user_task(self, plan):
        self.append_function_result_message("create_plan_to_complete_user_task", plan)
        return plan
