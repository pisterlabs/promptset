import ast
import datetime
import logging
from csv import DictWriter
from typing import List, Optional, Tuple, Union

import openai
import requests
import weaviate
from pydantic import BaseModel

from codinit.agents import (
    code_correcting_agent,
    coding_agent,
    dependency_agent,
    linting_agent,
    planner_agent,
)
from codinit.code_editor import PythonCodeEditor
from codinit.config import eval_settings

# from codinit.get_context import get_embedding_store, get_read_the_docs_context
from codinit.documentation.get_context import WeaviateDocQuerier
from codinit.documentation.pydantic_models import Library
from codinit.weaviate_client import get_weaviate_client

logger = logging.getLogger(__name__)
ANSWER_PATTERN = r"[a-zA-Z]+"
DEPENDENCY_BLACKLIST = set(["random", "json"])


def _trim_md(code_editor: PythonCodeEditor):
    if code_editor.source_code:
        code_editor.source_code[0] = code_editor.source_code[0].replace("```python", "")
        code_editor.source_code[-1] = code_editor.source_code[-1].replace("```", "")
        code_editor.overwrite_code(code_editor.display_code())


class TaskExecutionConfig(BaseModel):
    execute_code: bool = True
    install_dependencies: bool = True
    check_package_is_in_pypi: bool = True
    log_to_stdout: bool = True
    coding_attempts: int = 1
    max_coding_attempts: int = 5
    dependency_install_attempts: int = 5
    planner_temperature: float = 0
    coder_temperature: float = 0.0
    code_corrector_temperature: float = 0
    dependency_tracker_temperature: float = 0


class TaskExecutor:
    def __init__(
        self,
        code_editor: PythonCodeEditor,
        config: TaskExecutionConfig,
        task: str,
        run_id: int,
        task_id: int,
        sha: str,
        message: str,
        csv_writer: DictWriter,
    ) -> None:
        self.code_editor = code_editor
        self.config = config

        # Planner
        self.planner = planner_agent

        # Coder
        self.coder = coding_agent

        # Dependency tracker
        self.dependency_tracker = dependency_agent

        # Code corrector
        self.code_corrector = code_correcting_agent

        # linter
        self.linter = linting_agent

        self.task = task
        self.run_id = run_id
        self.task_id = task_id
        self.sha = (sha,)
        self.message = (message,)
        self.csv_writer = csv_writer

    def install_dependencies(self, deps: List[str]) -> str:
        # if it's a string, e.g. "['openai']", turn into list ['openai']
        if isinstance(deps, str):
            try:
                deps = ast.literal_eval(deps)
            except (SyntaxError, ValueError):
                print("The string couldn't be evaluated.")
        # print(type(deps))
        dependencies = []
        for d in deps:
            d = d.strip()
            if " " in d:
                d = d.split(" ")[0]

            if self.config.check_package_is_in_pypi:
                url = f"https://pypi.org/project/{d}"
                res = requests.get(url)
                if res.status_code != 200:
                    pass

            if len(d) < 2 or d in DEPENDENCY_BLACKLIST:
                continue

            dependencies.append(d)

        if dependencies:
            dependencies = list(set(dependencies))
            dependencies += ["langchain", "pydantic", "openai", "wikipedia"]
            logger.info(f"{dependencies=}")

            for dependency in dependencies:
                self.code_editor.add_dependency(dependency)

            self.code_editor.create_env()
            process = self.code_editor.install_dependencies()
            if process.returncode != 0:
                logger.error(f"Dependency install failed for: {dependencies}")

            message = f"dependency installer results: args={process.args}, return_code=stdout={process.stdout}, stderr={process.stderr}, return_code={process.returncode}"
        else:
            message = "no dependencies to install."
        return message

    def run_code(self, code: str) -> str:
        self.code_editor.overwrite_code(code)
        _trim_md(self.code_editor)
        self.code_editor.save_code()
        logger.info(self.code_editor.display_code())

        if not self.config.execute_code:
            return self.code_editor.display_code()

        result = self.code_editor.run_code()

        if "Succeeded" in result:
            logger.info("Source code is functional!")
            return "Task Success: " + result
        else:
            logger.info("Failed to generate an executable source code.")
            return "Task Failed: " + result

    def format_code(self, code: str, dependencies: List[str]) -> str:
        formatted_code = code
        for dep in dependencies:
            formatted_code = self.code_editor.process_imports(
                code=formatted_code, library_name=dep
            )
        return formatted_code

    """
    def get_docs_old(self, libraries: List[str], task: str):
        # first get context from provided libraries
        get_embedding_store(start_urls=libraries)
        relevant_docs = get_read_the_docs_context(task, k=5)
        return relevant_docs
    """

    def get_docs(self, library: Library, task: str, client: weaviate.Client):
        weaviate_doc_querier = WeaviateDocQuerier(library=library, client=client)
        docs = weaviate_doc_querier.get_relevant_documents(query=task)
        return docs

    def format_lint_code(
        self, code: str, dependencies: List[str]
    ) -> Tuple[str, List[str], int]:
        formatted_code = self.format_code(code=code, dependencies=dependencies)
        self.code_editor.overwrite_code(new_source=formatted_code)
        lint_result = (
            self.code_editor.run_linter()
        )  # validate_code_imports(code=new_code, dependencies = deps)
        metric = len(lint_result)
        # run generated code
        return formatted_code, lint_result, metric

    def write_row(
        self,
        attempt: int,
        formatted_code: str,
        lint_result: List[str],
        metric: int,
        error: Union[str, List[str]],
        time_stamp: str,
    ):
        row = [
            self.run_id,
            self.task_id,
            self.task,
            attempt,
            formatted_code,
            lint_result,
            metric,
            error,
            self.sha,
            self.message,
            time_stamp,
        ]
        row_dict = {
            key: value for key, value in list(zip(eval_settings.eval_columns, row))
        }
        self.csv_writer.writerow(row_dict)

    def code_correction_with_linting(
        self,
        new_code: str,
        deps: List[str],
        relevant_docs: str,
        attempt: int,
        time_stamp: str,
    ):
        formatted_code, lint_result, metric = self.format_lint_code(
            code=new_code, dependencies=deps
        )
        # feed in lint results
        print(f"{lint_result=}")
        # TODO: check if linting output is not empty
        if len(lint_result) > 0:
            lint_query_results = self.linter.execute(
                source_code=new_code, linter_output=lint_result
            )
            print(lint_query_results)
            lint_response = openai.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=self.linter.messages,
            )
            print(f"{lint_response=}")
            new_code = self.code_corrector.execute(
                tool_choice="execute_code",
                chat_history=[],
                task=self.task,
                context=relevant_docs,
                source_code=new_code,
                error=lint_response,
            )[0]
            formatted_code, lint_result, metric = self.format_lint_code(
                code=new_code, dependencies=deps
            )
        # run generated code
        error = self.run_code(formatted_code)
        # file has header: Run_ID,Task_ID,Task,Generation_ID,Code,Linter_Output,Metric,Error_Log,Git_SHA,Commit_Message,Timestamp
        self.write_row(
            attempt=attempt,
            formatted_code=formatted_code,
            lint_result=lint_result,
            metric=metric,
            error=error,
            time_stamp=time_stamp,
        )
        return error, new_code

    # TODO: add plan to benchmark
    def execute_and_log(
        self,
        library: Library,
        source_code: Optional[str] = None,
    ):
        client = get_weaviate_client()
        attempt = 0
        chat_history = []
        # Generating a coding plan
        time_stamp = datetime.datetime.now().isoformat()
        relevant_docs = self.get_docs(library=library, task=self.task, client=client)
        # generate coding plan given context
        plan = self.planner.execute(
            tool_choice="execute_plan",
            chat_history=[],
            task=self.task,
            context=relevant_docs,
        )[0]
        # install dependencies from plan
        if self.config.execute_code and self.config.install_dependencies:
            deps = self.dependency_tracker.execute(
                tool_choice="install_dependencies", chat_history=[], plan=plan
            )[0]
            self.install_dependencies(deps)
        chat_history.append(
            {"role": "assistant", "content": f"installed dependencies {deps}"}
        )
        # generate code
        new_code = self.coder.execute(
            task=self.task,
            tool_choice="execute_code",
            chat_history=chat_history,
            plan=plan,
            context=relevant_docs,
        )[0]
        error, new_code = self.code_correction_with_linting(
            new_code=new_code,
            deps=deps,
            relevant_docs=relevant_docs,
            attempt=attempt,
            time_stamp=time_stamp,
        )
        attempt = 1
        while "Failed" in error:
            if attempt > self.config.coding_attempts:
                break
            time_stamp = datetime.datetime.now().isoformat()
            # corrected code
            error, new_code = self.code_correction_with_linting(
                new_code=new_code,
                deps=deps,
                relevant_docs=relevant_docs,
                attempt=attempt,
                time_stamp=time_stamp,
            )
            attempt += 1
        return new_code
