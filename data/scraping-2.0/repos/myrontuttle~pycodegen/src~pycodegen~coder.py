from typing import Dict, List, Optional

import json
import logging
import os
import re
import subprocess
from json import JSONDecodeError
from pathlib import Path

import click
import tomli
from github.Issue import Issue
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import SimpleMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from pathvalidate import sanitize_filename

from pycodegen import llm, sc, tester, todo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


logger = logging.getLogger(__name__)

TEMP_FILE = "temp.py"
LOGGER_CODE = (
    "\nimport logging\n\n"
    "logging.basicConfig(\n"
    "    level=logging.INFO,\n"
    "    format='%(asctime)s [%(levelname)s] %(message)s',\n"
    "    datefmt='%Y-%m-%d %H:%M:%S'\n)\n\n"
    "logger = logging.getLogger(__name__)\n\n"
)
AI_COMMENT_TAG = "AI: "
NO_QUESTIONS = "No questions."


def review_comments(issue: Issue) -> int:
    """Review issue comments and take appropriate action"""
    comments = todo.get_issue_comments(issue)
    # Check if AI has already commented and gotten a response
    for idx, comment in enumerate(comments):
        if comment.startswith(AI_COMMENT_TAG):
            if len(comments) == idx + 1:  # Last comment
                if comment.find("?") == -1:
                    # AI has commented and has no questions
                    click.echo(f"No questions for issue #{issue.number}")
                    click.echo(f"Last comment: {comment}")
                    return 0
                # AI has commented, but needs a response
                click.echo(
                    "AI: Waiting for response from user in "
                    f"comments on issue #{issue.number}"
                )
                return 1
            else:
                # AI has commented and gotten a response
                return 0
    # AI has not commented yet
    for comment in comments:
        if comment.find("http://") != -1 or comment.find("https://") != -1:
            # TODO: Read linked page and act accordingly
            pass
    # AI ask questions about issue body and comments
    all_comments = "\n".join(comments)
    prompt = (
        f"What are the top questions that should be asked about the "
        f"following issue in order to develop an effective solution?\n"
        f"{issue.title}\n{issue.body}\n{all_comments}\n"
    )
    response = llm.complete_prompt(prompt)
    if response:
        todo.write_issue_comment(issue, AI_COMMENT_TAG + response)
        click.echo(f"Issue #{str(issue.number)}: {issue.title}")
        click.echo(AI_COMMENT_TAG + response)
        return 1
    else:
        logger.warning(f"No response from Chat LLM when asking:\n{prompt}")
        return 0


def bump_version(issue_type: str) -> None:
    """
    Bumps the version in pyproject.toml based on issue type
    Args:
        issue_type: The type of issue that was worked on

    Returns:
        None
    """
    if issue_type == "feature" or issue_type == "feat":
        logger.info("Bumping minor version for feature")
        cp_format = subprocess.run(
            [
                "pdm",
                "bump",
                "minor",
            ],
            capture_output=True,
        )
    else:
        logger.info(f"Bumping micro version for {issue_type}")
        cp_format = subprocess.run(
            [
                "pdm",
                "bump",
                "micro",
            ],
            capture_output=True,
        )
    if cp_format.returncode == 0:
        logger.info(cp_format.stdout)
    else:
        logger.error(cp_format.stderr)


def just_the_code(llm_text: str) -> Optional[str]:
    """
    Returns just the code portion of an LLM response.
    Args:
        llm_text

    Returns:
        Just the code portion of the text provided
    """
    if "```python" in llm_text:
        start_idx = llm_text.find("```python") + 10
        end_idx = llm_text.find("```", start_idx)
        return llm_text[start_idx:end_idx]
    elif "```" in llm_text:
        start_idx = llm_text.find("```") + 4
        end_idx = llm_text.find("```", start_idx)
        return llm_text[start_idx:end_idx]

    if "import " not in llm_text and "def " not in llm_text:
        # Probably not python code
        logger.warning("LLM response doesn't appear to be code: ")
        logger.warning(llm_text)
        return '"""' + llm_text + '"""'
    else:
        # We'll probably need to add more conditions as we encounter them
        return llm_text


def add_text_to_module(module_text: str, text_to_add: str) -> str:
    """
    Adds function text to module text in an organized way
    Args:
        module_text
        text_to_add

    Returns:
        Updated module text
    """
    # TODO: Add any imports to the top of the module
    class_loc = module_text.find("class ")
    main_loc = module_text.find('if __name__ == "__main__":')
    if class_loc != -1:
        end_of_class = module_text.find("\n\n\n", class_loc)
        if end_of_class == -1:
            # Nothing after last class function
            module_text += "\n\n" + text_to_add + "\n"
        else:
            # Adding text to end of class
            module_text = (
                f"{module_text[:end_of_class + 2]}"
                f"{text_to_add}"
                f"{module_text[end_of_class + 2:]}"
            )
    elif main_loc != -1:
        # No class, but main function needs to stay at end
        module_text = (
            f"{module_text[:main_loc]}"
            f"{text_to_add}\n\n\n"
            f"{module_text[main_loc:]}"
        )
    else:
        # No class or main function, just add to end
        module_text += "\n\n" + text_to_add + "\n"
    return module_text


def add_logging(script_content: str) -> str:
    """Adds code for a logger to a python script"""
    if script_content.find("import logging") != -1:
        logger.info("Logging already included")
        return script_content
    import_re = re.compile(r"^\s*import\s.*$", re.MULTILINE)
    from_import_re = re.compile(r"^\s*from\s.*$", re.MULTILINE)
    last_import_match = None
    for match in reversed(list(import_re.finditer(script_content))):
        if (
            last_import_match is None
            or match.start() > last_import_match.start()
        ):
            last_import_match = match
    for match in reversed(list(from_import_re.finditer(script_content))):
        if (
            last_import_match is None
            or match.start() > last_import_match.start()
        ):
            last_import_match = match
    if last_import_match is not None:
        last_import_line = last_import_match.end()
        return (
            script_content[:last_import_line]
            + LOGGER_CODE
            + script_content[last_import_line:]
        )
    else:
        return LOGGER_CODE + script_content


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


@cli.command()
@click.argument("repo_owner")
@click.argument("repo_name")
@click.option("-i", "--issue_num", type=int)
def start(repo_owner: str, repo_name: str, issue_num: Optional[int]) -> None:
    coder = Coder(repo_owner, repo_name)
    response = coder.open_issue(issue_num)
    if response == 0:
        click.echo("Successfully started issue")


@cli.command()
@click.argument("repo_owner")
@click.argument("repo_name")
def code(repo_owner: str, repo_name: str) -> None:
    coder = Coder(repo_owner, repo_name)
    response = coder.start_coding()
    if response == 0:
        click.echo("Successfully started coding")


@cli.command()
@click.argument("repo_owner")
@click.argument("repo_name")
@click.option(
    "-m",
    "--commit_msg",
    default="",
    help="Commit message. " "Generated automatically" " if not provided",
)
def finish(repo_owner: str, repo_name: str, commit_msg="") -> None:
    coder = Coder(repo_owner, repo_name)
    response = coder.finish_issue(commit_msg)
    if response == 0:
        click.echo("Successfully completed issue")


class Coder:
    """
    Coder Class
    """

    work_dir = Path("C:\\Users\\myron\\PycharmProjects")
    host = "https://github.com"

    def __init__(self, owner_name: str, repo_name: str):
        """
        Initializes a coder on a project repo
        Args:
            owner_name
            repo_name
        """
        self.repo_owner = owner_name
        self.repo_name = repo_name
        self.repo = sc.use_repo(self.work_dir, self.repo_name, self.repo_owner)
        self.repo_path = self.work_dir.joinpath(self.repo_name)
        tester.create_test_dirs(self.repo_path)

        venv_path = self.repo_path.joinpath(".venv")
        # Get python version from pyproject.toml
        pyproject_path = self.repo_path.joinpath("pyproject.toml")
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as pyproject:
                project_conf = tomli.load(pyproject)
                requires = project_conf["project"]["requires-python"]
                python_version = requires.replace(">", "").replace("=", "")
        else:
            python_version = "3.9"
        if not venv_path.exists():
            os.chdir(self.repo_path)
            cp_setup = subprocess.run(
                [
                    "pdm",
                    "venv",
                    "create",
                    f"{python_version}",
                    "-v",
                ],
                capture_output=True,
            )
            cp_setup2 = subprocess.run(
                [
                    "pdm",
                    "venv",
                    "activate",
                    "in-project",
                ],
                capture_output=True,
                shell=True,
            )  # nosec B602
            subprocess.run(
                [
                    "pdm",
                    "use",
                    f"{python_version}",
                    "-i",
                    "-f",
                    "-vv",
                ],
                capture_output=True,
            )
            subprocess.run(
                [
                    "make",
                    "install",
                ],
                capture_output=True,
            )
            if cp_setup2.returncode == 0:
                # May be too much to push to logs
                logger.info(cp_setup.stdout)
            else:
                logger.error(cp_setup.stderr)

    def open_issue(self, issue_num: Optional[int]) -> int:
        """
        Open an issue to work on. Either a specific issue or next available
        Args:
            issue_num

        Returns:
            response code
        """
        github_repo = todo.get_repo(self.repo_owner, self.repo_name)
        if issue_num:
            issue = todo.get_issue(github_repo, issue_num)
        else:
            issue = todo.get_next_issue(self.repo_owner, self.repo_name)
        if not issue:
            return 1
        logger.info(f"Working on issue {issue.number}: {issue.title}")

        # Review issue comments
        rc = review_comments(issue)
        if rc != 0:
            return rc

        # Pull repo
        if self.repo.active_branch.name == "main":
            self.repo.git.pull()
        else:
            self.repo.git.fetch()
        logger.info("Pulled repo")

        # Checkout git branch
        branch_name = todo.issue_title_to_branch_name(github_repo, issue)
        sc.use_branch(self.repo, branch_name)
        logger.info(f"Created branch {branch_name}")

        return 0

    def start_coding(self) -> int:
        """Start coding on an issue"""
        # Get issue from branch name
        branch_name = sc.get_active_branch_name(self.repo)
        if branch_name == "main":
            logger.error("Cannot code on main branch. Start an issue first")
            return 1
        github_repo = todo.get_repo(self.repo_owner, self.repo_name)
        issue_num = int(todo.issue_num_from_branch_name(branch_name))
        if not issue_num:
            logger.error(
                "Could not get issue number from branch name. "
                "Start an issue first"
            )
            return 1
        issue = todo.get_issue(github_repo, issue_num)
        issue_type = todo.get_issue_type(github_repo, issue)

        # Create functional test if new feature
        if issue_type == todo.feature_type:
            feature_path = tester.create_feature(self.repo_path, issue)
            logger.info(f"Created feature file {feature_path}")
            func_test_path = tester.create_step_defs(feature_path)
            if func_test_path:
                logger.info(f"Created functional test file {func_test_path}")

        # TODO: Account for multiple packages, no package, or package name
        #  different from repo name
        package_name = self.repo_name.replace("-", "_")

        # Recommend module to work with
        src_file_name = self.recommend_filename(issue, package_name)
        logger.info(f"Recommended source file {src_file_name}")

        # Create unit tests if bug or feature
        if issue_type == "bug" or issue_type == "feature":
            unit_tests = tester.create_unit_tests(
                src_file_name,
                issue.body,
                issue_type,
                package_name,
            )
            unit_test_path = tester.write_unit_tests_to_file(
                self.repo_path,
                src_file_name,
                just_the_code(unit_tests),
            )
            logger.info(f"Created/updated test file {unit_test_path}")
        else:
            unit_tests = ""
            logger.info(f"No unit tests created for issue_type={issue_type}")

        # Recommended libraries
        libs = self.recommend_libraries(issue)
        recommended_libs = list(libs.keys())

        # Start writing code for the issue
        self.write_src_code(
            issue,
            issue_type,
            src_file_name,
            recommended_libs,
            package_name,
            unit_tests,
        )

        return 0

    def finish_issue(self, commit_msg: str) -> int:
        """
        Formats, commits, merge, and push any work on active branch
        Args:
            commit_msg: Commit message
        Returns:
            Completion response code
        """
        # Make sure were in repo root dir
        os.chdir(self.repo_path)

        # Auto increment version based on issue
        branch_name = sc.get_active_branch_name(self.repo)
        issue_prefix = todo.issue_prefix_from_branch_name(branch_name)
        issue_type = todo.get_issue_type_from_prefix(issue_prefix)
        bump_version(issue_type)

        # Format code
        cp_format = subprocess.run(
            [
                "make",
                "format",
            ],
            capture_output=True,
        )
        if cp_format.returncode == 0:
            logger.info(cp_format.stdout)
        else:
            logger.error(cp_format.stderr)
            return 1
        sc.add_files(self.repo, ["."])

        # Make commit msg based on branch_name and work done
        branch_name = sc.get_active_branch_name(self.repo)
        if not commit_msg:
            commit_msg = sc.generate_commit_msg(self.repo, branch_name)
        else:
            issue_prefix = todo.issue_prefix_from_branch_name(branch_name)
            issue_number = todo.issue_num_from_branch_name(branch_name)
            commit_msg = sc.add_commit_message_info(
                commit_msg,
                issue_prefix,
                issue_number,
            )
        logger.info(f"Commit message: {commit_msg}")

        git_response_code = sc.commit(self.repo, commit_msg)
        if git_response_code != 0:
            return 1
        logger.info("Committed changes")
        git_response_code = sc.safe_merge(self.repo, branch_name)
        if git_response_code != 0:
            return 1
        logger.info("Merged changes")
        git_response_code = sc.push_to_origin(self.repo)
        if git_response_code != 0:
            return 1
        logger.info("Pushed changes")
        sc.delete_branch(self.repo, branch_name)
        logger.info(f"Deleted branch {branch_name}")
        return 0

    def recommend_libraries(self, issue: Issue) -> Optional[Dict[str, str]]:
        """
        Recommends a library based on an issue
        Args:
            issue

        Returns:
            Recommended library
        """
        # TODO: Consider adding project description for context in prompt
        # Ask Chat LLM what libraries it would recommend for issue
        prompt = (
            f"In the form of a JSON object, "
            f"what are the top python libraries I could use for the following "
            f"issue?\n{issue.title}\n{issue.body}\nRespond in the "
            f"form of a JSON object with each library name as "
            f"the key and a string of two sentences describing the "
            f"library and why to use it for this issue as the value."
        )
        response: str = llm.complete_prompt(prompt)
        if not response:
            return None

        click.echo(f"Recommended Libraries for issue #{str(issue.number)}:")
        click.echo(response)
        response = response[response.find("{") : response.find("}") + 1]
        if response.find("': ") != -1:
            response = response.replace("'", '"')
        try:
            recommendations = json.loads(response)
        except JSONDecodeError as jde:
            logger.warning("Can't load LLM response as JSON. " + str(jde))
            logger.warning("Response: " + response)
            return None

        # Lookup alternatives
        rec_list = " or ".join(recommendations.keys())
        # TODO: Do this with a web search to get current best practices
        alt_prompt = (
            f"In the form of a JSON object, what are some "
            f"alternative python libraries to using {rec_list}? "
            f"Respond in the form of a JSON object with each "
            f"library name as the key and a string of two "
            f"sentences describing the library and why to use it "
            f"for this ticket as the value."
        )
        alt_response: str = llm.complete_prompt(alt_prompt)
        if alt_response:
            alt_response = alt_response.replace("'", '"')
            alt_response = alt_response[
                alt_response.find("{") : alt_response.find("}") + 1
            ]
            try:
                recommendations.update(json.loads(alt_response))
            except JSONDecodeError as jde:
                logger.error(str(jde))
                logger.debug(
                    "Unable to get library alternatives.\n"
                    f"Messages: {str(alt_prompt)}\n"
                    f"Response: {alt_response}"
                )

        # If the project is already using the libraries, recommend those
        to_recommend = {}
        with open(self.repo_path.joinpath("pyproject.toml"), mode="rb") as fp:
            project_config = tomli.load(fp)
        dependencies = project_config["project"]["dependencies"]
        for rec in recommendations.keys():
            if rec in dependencies:
                to_recommend[rec] = recommendations[rec]
        if to_recommend:
            return to_recommend
        else:
            return recommendations

    def add_library(self, lib_name: str) -> None:
        """
        Adds a library to the project dependencies using PDM if it's not
        already a dependency
        """
        os.chdir(self.repo_path)
        with open(self.repo_path.joinpath("pyproject.toml"), mode="rb") as fp:
            project_config = tomli.load(fp)
        dependencies = project_config["project"]["dependencies"]
        if lib_name in dependencies:
            logger.info(f"Library: {lib_name} already in project dependencies")
            return None
        cp_add_lib = subprocess.run(
            [
                "pdm",
                "add",
                f"{lib_name}",
            ],
            capture_output=True,
        )
        if cp_add_lib.returncode == 0:
            logger.info(cp_add_lib.stdout)
        else:
            logger.error(cp_add_lib.stderr)

    def recommend_filename(self, issue: Issue, pkg_name="") -> str:
        """
        Recommend a filename to create or add to for the issue
        Args:
            issue: Issue
            pkg_name: Package name

        Returns:
            Filename
        """
        # Get existing src file
        if not pkg_name:
            # Assume package name same as repo name
            pkg_name = self.repo_name.replace("-", "_")
        src_dir = self.repo_path.joinpath("src").joinpath(pkg_name)
        src_files = [
            str(f)
            for f in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, f))
            and str(f).endswith(".py")
            and str(f) != "__init__.py"
        ]
        # Ask Chat LLM what filename it would recommend for issue
        if src_files:
            prompt = (
                f"Recommend one of these source files ("
                f"{', '.join(src_files)}) to add "
                f"python code that would solve the below issue. "
                f"If none of the source files is appropriate, respond with "
                f"a new file name.\n"
                f"Issue: {issue.title}\n{issue.body}\n"
                f"Respond with just the name of the file."
            )
        else:
            prompt = (
                f"What should I name the python script that solves the "
                f"following issue?\n"
                f"Issue: {issue.title}\n{issue.body}\n"
                f"Respond with just the name of the file."
            )

        response: str = llm.complete_prompt(prompt)
        if response:
            if response.find(" ") != -1:
                response = response[response.rfind(" ") + 1 :]
            response = sanitize_filename(response)
            if not response.endswith(".py"):
                response += ".py"
            return response
        else:
            logger.error(
                f"No file name recommended for issue #: {issue.number}."
                f"Using {TEMP_FILE}."
            )
            return TEMP_FILE

    def write_src_code(
        self,
        issue: Issue,
        issue_type: str,
        src_file_name: str,
        recommended_libs: List[str],
        package_name="",
        unit_tests="",
    ) -> None:
        """
        Writes code to the file provided (creating if it doesn't exist)
        Args:
            issue: Issue
            issue_type: Issue type
            src_file_name: Source file name
            recommended_libs: Recommended libraries
            package_name: Package name
            unit_tests: Unit tests

        Returns:
            None
        """
        chat = ChatOpenAI(model_name=llm.CHAT_MODEL)
        role_template = "You are a thoughtful python software developer."
        # Plan steps for issue solution
        # TODO: Consider adding project description for context in prompt
        steps_template = (
            "{role} Evaluate the following {issue_type} and comments. "
            "Return a list of just the steps you would take to solve it.\n"
            "{issue_type}: {issue_body}\n\nComments:\n{comments}"
        )
        steps_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate.from_template(steps_template)
        )
        steps_chat = ChatPromptTemplate.from_messages([steps_prompt])
        steps_chain = LLMChain(
            llm=chat,
            prompt=steps_chat,
            output_key="solution_steps",
        )
        # Critique steps
        critique_template = (
            "{role} Compare the following list of steps with the "
            "{issue_type} and comments. Which steps are redundant, incorrect, "
            "or can be simplified?\n{issue_type}: {issue_body}\n\n"
            "Comments:\n{comments}\n"
            "Steps:\n{solution_steps}"
        )
        critique_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate.from_template(critique_template)
        )
        critique_chat = ChatPromptTemplate.from_messages([critique_prompt])
        critique_chain = LLMChain(
            llm=chat,
            prompt=critique_chat,
            output_key="critique",
        )
        # Write updated steps
        update_template = (
            "{role}\n"
            "For a solution to this {issue_type}:\n{issue_body}\n"
            "With comments:\n{comments}\n"
            "You wrote steps:\n{solution_steps}"
            "You critiqued the steps:\n{critique}\n\n"
            "Write what the steps should be (if there are no "
            "changes needed, please repeat the steps)."
        )
        update_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate.from_template(update_template)
        )
        update_chat = ChatPromptTemplate.from_messages([update_prompt])
        update_chain = LLMChain(
            llm=chat,
            prompt=update_chat,
            output_key="updated_steps",
        )
        # Write source code
        source_template = (
            "{role} Write python code for the "
            "following steps:\n{updated_steps}\n"
            "You can use the following libraries:\n{use_libs}\n"
            "The code produced should pass the following unit tests.\n"
            "{unit_tests}\n"
            "Respond with just the python code.\n"
        )
        source_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate.from_template(source_template)
        )
        source_chat = ChatPromptTemplate.from_messages([source_prompt])
        source_chain = LLMChain(
            llm=chat,
            prompt=source_chat,
            output_key="source_code",
        )
        create_source_chain = SequentialChain(
            memory=SimpleMemory(memories={"role": role_template}),
            chains=[
                steps_chain,
                critique_chain,
                update_chain,
                source_chain,
            ],
            input_variables=[
                "issue_type",
                "issue_body",
                "comments",
                "use_libs",
                "unit_tests",
            ],
            output_variables=[
                "solution_steps",
                "critique",
                "updated_steps",
                "source_code",
            ],
            verbose=True,
        )
        # Get comments
        comments = "\n".join(todo.get_issue_comments(issue))
        # Get existing imports
        src_file_path = self.repo_path.joinpath(
            "src", package_name, src_file_name
        )
        src_file_contents = ""
        if src_file_path.exists():
            with open(src_file_path, "r") as fp:
                src_file_contents = fp.read()
        imports = []
        if src_file_contents:
            # Get existing imports in file
            imports = [
                line
                for line in src_file_contents.split("\n")
                if line.startswith("import") or line.startswith("from")
            ]
        # Add recommended libraries to imports
        for lib in recommended_libs:
            import_lib = f"import {lib}"
            if import_lib not in imports:
                imports.append(import_lib)
        use_libs = "\n".join(imports)

        result = create_source_chain(
            {
                "issue_type": issue_type,
                "issue_body": issue.body,
                "comments": comments,
                "use_libs": use_libs,
                "unit_tests": unit_tests,
            }
        )
        # Write source code to file
        response = result["source_code"]
        if response:
            if src_file_contents:
                src_file_contents = add_text_to_module(
                    src_file_contents,
                    just_the_code(response),
                )
            else:
                src_file_contents = just_the_code(response)
                src_file_contents = add_logging(src_file_contents)
            with open(src_file_path, "w") as fp:
                fp.write(src_file_contents.replace("\r", ""))
            logger.info(f"Added the following to file {src_file_path}")
            logger.info(f"{src_file_contents}")
        else:
            logger.warning("No response from LLM")
            logger.warning("No source code written.")


if __name__ == "__main__":
    cli()
