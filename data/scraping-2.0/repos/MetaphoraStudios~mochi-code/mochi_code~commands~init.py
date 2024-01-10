"""The init command. This command is used to initialize Mochi for a new 
project."""

import argparse
import pathlib

from dotenv import dotenv_values
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.output_parsers import (CommaSeparatedListOutputParser,
                                      PydanticOutputParser)
from retry import retry

from mochi_code.code import ProjectDetails, ProjectDetailsWithDependencies
from mochi_code.code.mochi_config import create_config, search_mochi_config
from mochi_code.commands.exceptions import MochiCannotContinue

# Load keys for the different model backends. This needs to be setup separately.
keys = dotenv_values(".keys")


def setup_init_arguments(parser: argparse.ArgumentParser) -> None:
    """Setup the arguments for the init command.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments to.
    """
    parser.add_argument("-f",
                        "--force",
                        action="store_true",
                        help="Force creating the config, without overriding.")


def run_init_command(args: argparse.Namespace) -> None:
    """Run the init command with the provided arguments."""
    # Arguments should be validated by the parser.
    project_path = pathlib.Path.cwd()
    existing_root = search_mochi_config(project_path)

    if not args.force and existing_root is not None:
        raise MochiCannotContinue(
            f"🚫 Mochi is already initialized at '{existing_root.parent}'.")

    if existing_root is not None and existing_root.parent == project_path:
        print("😎 Mochi already exists in this folder, left intact.")
        return

    init(project_path)


def init(project_path: pathlib.Path) -> None:
    """Run the init command.
    
    Args:
        project_path (pathlib.Path): The path to the project to initialize (the
        config folder will be created here).
    """
    print(f"⚙️  Initializing mochi for project '{project_path}'.")
    print("🤖 Gathering information about your project...")

    project_files = [p.name for p in project_path.glob("*")]
    project_details = _get_project_details(project_files)

    print("🤖 Gathering list of dependencies...")
    dependencies = _get_dependencies_list(project_details)
    complete_project_details = ProjectDetailsWithDependencies(
        **project_details.dict(), dependencies=dependencies)

    config_path = create_config(project_path, complete_project_details)

    config_display_uri = config_path.relative_to(project_path).as_posix()
    print(f"🤖 Created the config at {config_display_uri}")


@retry(tries=3)
def _get_project_details(project_files: list[str]) -> ProjectDetails:
    """Get the details of a project from the user.

    Returns:
        ProjectDetails: The details of the project.
    """
    llm = OpenAI(temperature=0.5,
                 openai_api_key=keys["OPENAI_API_KEY"])  # type: ignore

    parser = PydanticOutputParser(pydantic_object=ProjectDetails,)
    template = PromptTemplate(
        input_variables=["files"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
        template="You are a professional software engineer helping other " +
        "engineers. I'm going to provide a comma separated list of " +
        "files from a project and you need to reply with some " +
        "information about the project in a valid single line JSON " +
        "format with lower case values.\n{format_instructions}\nOutput must " +
        "be a valid json!\nHere's the list of files:\n{files}",
    )
    chain = LLMChain(llm=llm, prompt=template)

    response = chain.run(files=",".join(project_files))

    return parser.parse(response)


def _get_dependencies_list(project_details: ProjectDetails) -> list[str]:
    """Get the list of dependencies from the dependencies file.

    Args:
        project_details (ProjectDetails): The details of the project we're
        extracting the dependencies from.
    
    Returns:
        list[str]: The list of dependencies or empty if none could be found.
    """
    if not project_details.config_file:
        # Avoids the case where it is treated as the "." file.
        return []

    dependencies_config_path = pathlib.Path(project_details.config_file)
    if not dependencies_config_path.exists():
        return []

    dependencies_config_content = _load_dependencies_config_content(
        dependencies_config_path)
    return _fetch_list_of_dependencies(project_details.language,
                                       project_details.package_manager,
                                       dependencies_config_content)


def _load_dependencies_config_content(
        dependencies_config_path: pathlib.Path) -> str:
    """Load the dependencies config file.

    Args:
        dependencies_config_path (pathlib.Path): The path to the config file
        defining the dependencies.
    
    Returns:
        str: The dependencies config file content.
    """
    with open(dependencies_config_path, encoding="utf-8") as config_file:
        return config_file.read()


@retry(tries=3)
def _fetch_list_of_dependencies(language: str, package_manager: str,
                                dependencies_config_content: str) -> list[str]:
    """Fetch the list of dependencies from the modal.

    Returns:
        list[str]: The list of dependencies or empty if none could be found.
    """
    llm = OpenAI(temperature=0.5,
                 openai_api_key=keys["OPENAI_API_KEY"])  # type: ignore

    parser = CommaSeparatedListOutputParser()
    template = PromptTemplate(
        input_variables=["language", "package_manager", "config_content"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
        template="I am going to provide the contents of a config file for a " +
        "{language} project using {package_manager} package manager.\nI " +
        "would like you to output the list of dependencies required to run " +
        "the project.\n{format_instructions}\nThe config file:\n```\n" +
        "{config_content}\n```",
    )
    chain = LLMChain(llm=llm, prompt=template)

    response = chain.run(language=language,
                         package_manager=package_manager,
                         config_content=dependencies_config_content)

    return parser.parse(response)
