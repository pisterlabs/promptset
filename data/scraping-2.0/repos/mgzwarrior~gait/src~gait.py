import json
import logging
import os
import readline
import subprocess
from getpass import getpass

import click
import click_config_file

from services.constants import CONFIG_FILENAME, ENV_FILENAME, GAIT_LOGNAME
from services.exceptions import GitException, GitHubException, OpenAIException
from services.git import GitService
from services.github import GitHubService
from services.openai import OpenAIService

logger = logging.getLogger(GAIT_LOGNAME)
logging.basicConfig(
    filename=f"../{GAIT_LOGNAME}.log", filemode="w", level=logging.DEBUG
)


@click.group()
def gait() -> None:
    """Gait is a CLI tool that uses OpenAI's ChatGPT to generate commit messages.
    It is designed to be used with Git.
    """
    return None


@gait.command()
@click.option(
    "--auto", "-a", default=False, help="Automatic commit mode.", is_flag=True
)
@click.option("--message", "-m", default="", help="Commit message.")
@click.option(
    "--skip", "-s", default=False, help="Skip OpenAI message generation.", is_flag=True
)
@click.option(
    "--track",
    "-t",
    default=False,
    help="Track commit flow to train the OpenAI model",
    is_flag=True,
)
@click.option("--verbose", "-v", default=False, help="Verbose mode.", is_flag=True)
@click_config_file.configuration_option(
    config_file_name=CONFIG_FILENAME
)  # Note that this does not work implicitly
def commit(auto, message, skip, track, verbose) -> None:
    """This command is ued to generate a commit message using ChatGPT.
    The message is generated based on the diff of the current branch and the master branch.
    There are two modes for this command: interactive mode (default) and automatic mode.
    """
    print("Beginning gait commit...")

    git_service = GitService()
    openai_service = OpenAIService()

    if message != "":
        skip = True

    if not skip:
        try:
            diff_or_diff_fn = git_service.diff(track)
        except GitException as exc:
            logger.error(exc)
            raise click.ClickException(str(exc))

        try:
            if track:
                with open(diff_or_diff_fn, "r", encoding="utf-8") as diff_file:
                    diff = diff_file.read()
                    commit_message = json.loads(
                        openai_service.generate_commit_message(diff)
                    )

                    if verbose:
                        print(f"Diff: {diff}")
                        print(
                            f"Generated commit message: {json.dumps(commit_message, indent=4)}"
                        )
            else:
                commit_message = json.loads(
                    openai_service.generate_commit_message(diff_or_diff_fn)
                )

                if verbose:
                    print(f"Diff: {diff_or_diff_fn}")
                    print(
                        f"Generated commit message: {json.dumps(commit_message, indent=4)}"
                    )

                logger.info(json.dumps(commit_message, indent=4))
        except OpenAIException as exc:
            logger.error(exc)
            raise click.ClickException(str(exc))

        message = json.dumps(commit_message["choices"][0]["text"], indent=4)

    if auto:
        __git_commit(git_service, message)
    else:
        choice = "edit"

        if not skip:
            print(f"ChatGPT generated the following commit message: '{message}'")
            print("Would you like to commit this message? [y/n/edit]")

            choice = input()

        if choice == "y":
            __git_commit(git_service, message)
        elif choice == "edit":
            print("Please enter your commit message below:")
            user_commit_message = __rlinput(message)
            __git_commit(git_service, user_commit_message)
        else:
            print("Aborting...")
            return

    print("Gait commit complete!")


@gait.command()
@click.option("--verbose", "-v", default=False, help="Verbose mode.", is_flag=True)
@click_config_file.configuration_option(
    config_file_name=CONFIG_FILENAME
)  # Note that this does not work implicitly
def configure(verbose) -> None:
    """This command is used to configure Gait.
    If required, it will prompt the user for their OpenAI API key and test the connection.
    """
    print("Setting up Gait...")

    if os.getenv("OPENAI_API_KEY"):
        __test_openai_connection(verbose)
    else:
        print(
            "In order to use Gait, you must setup an OpenAI API key for your account."
        )
        print(
            "Navigating to https://platform.openai.com/account/api-keys to create a new key."
        )

        key = getpass(prompt="Please enter your OpenAI API Key: ")

        with open(ENV_FILENAME, "w") as env_file:
            env_file.write(f"{key}\n")

        __test_openai_connection(verbose)

    if not __check_for_gh_cli():
        __install_gh_cli()

    print("Gait setup complete!")


@gait.command()
@click.option(
    "--auto", "-a", default=False, help="Automatic commit mode.", is_flag=True
)
@click.option("--verbose", "-v", default=False, help="Verbose mode.", is_flag=True)
@click_config_file.configuration_option(
    config_file_name=CONFIG_FILENAME
)  # Note that this does not work implicitly
def push(auto, verbose) -> None:
    """This command is ued to push changes to the remote repository and create a pull request
    with a title and description generated using ChatGPT.
    The title and description are generated based on the commits being pushed to the remote.
    There are two modes for this command: interactive mode (default) and automatic mode.
    """
    # TODO: add verbose logging
    print("Beginning gait push...")

    git_service = GitService()
    openai_service = OpenAIService()
    github_service = GitHubService()

    if auto:
        __git_push(git_service)
        title = openai_service.generate_pull_request_title()
        description = openai_service.generate_pull_request_description()
        __gh_create_pull_request(github_service, title, description)
    else:
        # TODO: add check for uncommitted changes
        # TODO: add commit info to printout
        print("You have the following commits ready to push.  Continue? [y/n]")

        choice = input()

        if choice == "y":
            __git_push(git_service)
        else:
            print("Aborting...")

        title = openai_service.generate_pull_request_title()
        description = openai_service.generate_pull_request_description()

        print(
            f"ChatGPT generated the following pull request title: '{title}' and description: '{description}'"
        )
        print(
            "Would you like to create pull request using this title and description? [y/n/edit]"
        )

        choice = input()

        if choice == "y":
            __gh_create_pull_request(github_service, title, description)
        elif choice == "edit":
            print("Please enter your pull request title below:")
            user_title = input()
            print("Please enter your pull request description below:")
            user_description = input()
            __gh_create_pull_request(github_service, user_title, user_description)
        else:
            print("Aborting...")
            return

    print("Gait push complete!")


def __check_for_gh_cli() -> bool:
    print("Checking for GitHub CLI...")
    cmd = ["gh --version"]

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(exc)
        return False

    print("GitHub CLI found!")

    return True


def __gh_create_pull_request(
    service: GitHubService, title: str, description: str
) -> None:
    print("Creating pull request using GitHub CLI...")

    try:
        service.create_pull_request(title, description)
    except GitHubException as exc:
        logger.error(exc)
        raise click.ClickException(str(exc))

    print("Pull request created!")


def __git_commit(service: GitService, message: str) -> None:
    print("Committing...")

    try:
        service.commit(message)
    except GitException as exc:
        logger.error(exc)
        raise click.ClickException(str(exc))

    print("Commit successful!")


def __git_push(service: GitService) -> None:
    print("Pushing...")

    try:
        service.push()
    except GitException as exc:
        logger.error(exc)
        raise click.ClickException(str(exc))


def __install_gh_cli() -> None:
    print("Installing GitHub CLI...")
    cmd = ["brew install gh"]

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(exc)
        raise click.ClickException(str(exc))

    print("GitHub CLI installed!")


def __rlinput(message: str) -> str:
    readline.set_startup_hook(lambda: readline.insert_text(message))
    try:
        return input()
    finally:
        readline.set_startup_hook()


def __test_openai_connection(verbose: bool) -> None:
    print("Testing OpenAI connection...")

    try:
        openai_service = OpenAIService()
        response = openai_service.test_connection()
    except OpenAIException as exc:
        logger.error(exc)
        raise click.ClickException(str(exc))

    if verbose:
        print(f"OpenAI response: {json.dumps(response, indent=4)}")

    print("OpenAI setup complete!")


if __name__ == "__main__":
    gait()
