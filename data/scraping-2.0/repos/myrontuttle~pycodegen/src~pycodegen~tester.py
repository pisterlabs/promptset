from typing import List, Optional

import logging
import os
import subprocess
from pathlib import Path

from github.Issue import Issue
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import SimpleMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from pycodegen import llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

tests_dir = "tests"
features_dir = "features"
step_def_dir = "functional"
unit_dir = "unit"


def create_test_dirs(repo_path: Path) -> None:
    """
    Add directories for features, step definitions, and unit tests if they
    don't exist
    Args:
        repo_path

    Returns:
        None
    """
    tests_path = repo_path.joinpath(tests_dir)
    tests_path.joinpath(features_dir).mkdir(parents=True, exist_ok=True)
    tests_path.joinpath(step_def_dir).mkdir(parents=True, exist_ok=True)
    tests_path.joinpath(unit_dir).mkdir(parents=True, exist_ok=True)


def create_feature(repo_path: Path, issue: Issue) -> Path:
    """
    Create feature file from issue
    Args:
        repo_path
        issue

    Returns:
        Feature file path
    """
    feature_name = issue.title.replace(" ", "_") + ".feature"
    feature_file_path = (
        repo_path.joinpath(tests_dir)
        .joinpath(features_dir)
        .joinpath(feature_name)
    )
    indented_issue_body = indent_sub_lines(issue.body)

    if os.path.exists(feature_file_path):
        logger.warning(
            f"Feature file {feature_file_path} already exists. "
            f"Adding scenario(s)."
        )
        with open(feature_file_path, "r") as ffp:
            existing_scenarios = get_scenarios_from_issue_body(ffp.read())
        scenarios = get_scenarios_from_issue_body(indented_issue_body)
        for scenario in scenarios:
            if scenario in existing_scenarios:
                scenarios.remove(scenario)
        if scenarios:
            with open(feature_file_path, "a") as fp:
                fp.write("\n\n  ")
                fp.writelines(scenarios)
        else:
            logger.info("No new scenarios to add.")
    else:
        with open(feature_file_path, "w") as fp:
            fp.write("Feature: " + issue.title + "\n")
            fp.write(indented_issue_body)
    return feature_file_path


def get_scenarios_from_issue_body(issue_body: str) -> List[str]:
    """
    Returns a list of scenarios from the issue body
    Args:
        issue_body

    Returns:
        List of scenarios
    """
    scenarios = []
    scenario_loc = issue_body.find("cenario:")
    if scenario_loc == -1:
        logger.warning(
            "No scenarios found in issue body. Looking for " "'cenario:'."
        )
        return scenarios
    scenario_loc -= 1
    next_scenario_loc = issue_body.find("cenario:", scenario_loc + 9)
    while next_scenario_loc != -1:
        next_scenario_loc -= 1
        scenarios.append(issue_body[scenario_loc:next_scenario_loc])
        scenario_loc = next_scenario_loc
    scenarios.append(issue_body[scenario_loc:])
    return scenarios


def indent_sub_lines(issue_body: str) -> str:
    """
    Indent lines in the issue body that don't start with Feature or Scenario
    Args:
        issue_body

    Returns:
        Issue body with sub lines indented
    """
    # Remove all carriage returns and add back in only "\n"
    issue_lines = issue_body.splitlines(False)
    saw_scenario = False
    for idx, line in enumerate(issue_lines):
        if (
            line.startswith("Feature:")
            or line.startswith("feature:")
            or line.isspace()
            or not line
        ):
            issue_lines[idx] = line + "\n"
            continue
        if line.startswith("Scenario:") or line.startswith("scenario:"):
            issue_lines[idx] = "  " + line + "\n"
            saw_scenario = True
            continue
        if not saw_scenario:
            issue_lines[idx] = "  " + line + "\n"
        else:
            if (
                line.startswith("Given")
                or line.startswith("given")
                or line.startswith("When")
                or line.startswith("when")
                or line.startswith("Then")
                or line.startswith("then")
                or line.startswith("And")
                or line.startswith("and")
            ):
                issue_lines[idx] = "    " + line + "\n"
            else:
                issue_lines[idx] = "      " + line + "\n"

    return "".join(issue_lines)


def create_step_defs(feature_path: Path) -> Optional[Path]:
    """
    Create definition file from feature file
    Args:
        feature_path

    Returns:
        step_def file path
    """
    feat_file_name = os.path.split(feature_path)[1]
    feature_name = feat_file_name[: feat_file_name.find(".")]
    test_filename = "test_" + feature_name + ".py"
    test_root = feature_path.parent.joinpath("..")
    test_path = test_root.joinpath(step_def_dir).joinpath(test_filename)
    try:
        cp_step_def = subprocess.run(
            [
                "pytest-bdd",
                "generate",
                f"{feature_path}",
            ],
            capture_output=True,
        )
        if cp_step_def.returncode == 0:
            with open(test_path, "w") as tp:
                test_body = cp_step_def.stdout.decode("UTF-8", errors="ignore")
                tp.write(test_body.replace("\r", ""))
            fix_step_def_functions(test_path)
            return test_path
        else:
            logger.error(cp_step_def.stderr)
    except FileNotFoundError as err:
        logger.error(err)
        logger.error(f"Feature file {feature_path} not found.")


def fix_step_def_functions(test_path: Path) -> None:
    """
    Write titles for step def functions
    Args:
        test_path

    Returns:
        None
    """
    with open(test_path, "r") as tp:
        test_lines = tp.readlines()
    test_lines[0].lstrip('"').rstrip('"')
    prompt_base = (
        "Write just the python function title for the following "
        "step def:\n "
    )
    for idx, line in enumerate(test_lines):
        if line.startswith("@scenario('features\\"):
            test_lines[idx] = line.replace(
                "@scenario('features\\", "@scenario('../features/"
            )
        if line.startswith('@scenario("features\\'):
            test_lines[idx] = line.replace(
                '@scenario("features\\', '@scenario("../features/'
            )
        if line.startswith("def _():"):
            step_def = (
                test_lines[idx - 1]
                .replace("@given(", "Given ")
                .replace("@when(", "When ")
                .replace("@then(", "Then ")
                .replace(")", "")
            )
            prompt = prompt_base + step_def
            response = llm.complete_prompt(prompt=prompt)
            if response.find(" ") != -1:
                response = response[response.rfind(" ") + 1 :]
            if response:
                response = (
                    response.replace("def ", "")
                    .replace("()", "")
                    .replace(":", "")
                )
                test_lines[idx] = line.replace("_", response)

    with open(test_path, "w") as tp:
        tp.writelines(test_lines)


def create_unit_tests(
    src_file_name: str,
    issue_body: str,
    issue_type: str,
    package_name: str,
) -> Optional[str]:
    """
    Create unit tests from issue description
    Args:
        src_file_name: Source file name
        issue_body: Issue description
        issue_type: Issue type
        package_name: Package name

    Returns:
        unit test file path
    """
    source_module = src_file_name[: src_file_name.find(".")]
    chat = ChatOpenAI(model_name=llm.CHAT_MODEL)
    role_template = (
        "You are a great QA engineer preparing a suite of unit "
        "tests for Test Driven Development."
    )
    # Create test cases
    test_case_template = (
        "{role} Evaluate the following {issue_type} and "
        "return a list of all of the test cases that "
        "should be tested.\nIssue: {issue_body}"
    )
    test_case_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate.from_template(test_case_template)
    )
    test_case_chat = ChatPromptTemplate.from_messages([test_case_prompt])
    test_case_chain = LLMChain(
        llm=chat,
        prompt=test_case_chat,
        output_key="test_cases",
    )
    # Critique test cases
    critique_template = (
        "{role} Compare the list of test cases with the "
        "{issue_type}. Which of the test "
        "cases are redundant, incorrect, or can be "
        "simplified?\nIssue: {issue_body}\n"
        "Test cases:\n{test_cases}"
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
    # Write updated test cases
    update_template = (
        "{role}\n"
        "For {issue_type}: {issue_body}\n"
        "You wrote test cases:\n{test_cases}"
        "You critiqued the test cases:\n{critique}"
        "Write what the test cases should be (if there are no "
        "changes needed, please repeat the test cases):\n"
    )
    update_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate.from_template(update_template)
    )
    update_chat = ChatPromptTemplate.from_messages([update_prompt])
    update_chain = LLMChain(
        llm=chat,
        prompt=update_chat,
        output_key="updated_test_cases",
    )
    # Write unit tests
    unit_test_template = (
        "{role} Write python pytest unit tests for the "
        "following test cases:\n{updated_test_cases}\n"
        "Good unit tests should be independent, "
        "deterministic, self-validating, "
        "isolated, reproducible, and take advantage of the "
        "features of pytest to make the tests easy to "
        "write and maintain. Any explanation for the tests should be made "
        "as comments in python. For the module under test use the name "
        "'{source_module}' in the '{package_name}' package (for example: "
        "from {package_name} import {source_module}). Return just the "
        "python code for the unit tests."
    )
    unit_test_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate.from_template(unit_test_template)
    )
    unit_test_chat = ChatPromptTemplate.from_messages([unit_test_prompt])
    unit_test_chain = LLMChain(
        llm=chat,
        prompt=unit_test_chat,
        output_key="unit_tests",
    )
    create_tests_chain = SequentialChain(
        memory=SimpleMemory(memories={"role": role_template}),
        chains=[
            test_case_chain,
            critique_chain,
            update_chain,
            unit_test_chain,
        ],
        input_variables=[
            "issue_body",
            "issue_type",
            "source_module",
            "package_name",
        ],
        output_variables=[
            "test_cases",
            "critique",
            "updated_test_cases",
            "unit_tests",
        ],
        verbose=True,
    )
    result = create_tests_chain(
        {
            "issue_body": issue_body,
            "issue_type": issue_type,
            "source_module": source_module,
            "package_name": package_name,
        }
    )
    return result["unit_tests"]


def write_unit_tests_to_file(
    repo_path: Path,
    src_file_name: str,
    unit_tests: str,
) -> Path:
    """
    Write unit tests to file
    Args:
        repo_path:
        src_file_name:
        unit_tests:

    Returns:

    """

    # Write result to test file
    test_file_name = "test_" + src_file_name
    test_file_path = (
        repo_path.joinpath(tests_dir)
        .joinpath(unit_dir)
        .joinpath(test_file_name)
    )

    if os.path.exists(test_file_path):
        logger.warning(
            f"Test file {test_file_path} already exists. Adding test(s)."
        )
        with open(test_file_path, "a") as fp:
            fp.write("\n\n")
            # Write lines that don't begin with import or from
            for line in unit_tests.split("\n"):
                if not line.startswith("import") and not line.startswith(
                    "from"
                ):
                    fp.write("\n" + line)
            fp.write("\n\n")
    else:
        with open(test_file_path, "w") as fp:
            fp.write(unit_tests)
    return test_file_path
