import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import openai
import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

OPENAI_GPT_TYPES = {
    "gpt-3.5-turbo",    # ChatGPT (this is always updated if they have a new version)
    "gpt-3.5-turbo-16k",    # ChatGPT with 16k token context --> apply this if the diff is too big (more expensive, ~2X the price of the normal one)
    "gpt-3.5-turbo-0301",    # Older ChatGPT released in March 2022
    "gpt-4",    # GPT-4 (this is more than ~10X the price compared to ChatGPT)
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prometh Review - AI Pull Request Reviewer Companion")
    parser.add_argument("--base-url", type=str)
    parser.add_argument("-p", "--project-key", type=str, help="Project key")
    # TODO: include the choices for the GPT types
    parser.add_argument("--llm-type", default="gpt-3.5-turbo", type=str, help="LLM type. Default: gpt-3.5-turbo")
    parser.add_argument(
        "-c",
        "--nb-context-lines",
        default=10,
        type=int,
        help=
        "how much lines to include before and after a changed line? Reduce it if you have problems with the script. Default: 10"
    )
    # TODO: repo slug could be extracted with a git command
    parser.add_argument("-r", "--repo", required=True, type=str, help="Slug of the repo")
    parser.add_argument("-id", "--pull-request-id", required=True, type=int, help="ID for the pull request")
    parser.add_argument("-d",
                        "--show-diff-only",
                        action="store_true",
                        help="Show diff only, no AI checks, used mainly for debugging")
    parser.add_argument("-i",
                        "--interactive",
                        action="store_true",
                        help="You can continue to chat about the PR with the AI")
    parser.add_argument(
        "-e",
        "--exhaustive-analysis",
        action="store_true",
        help=
        "Perform an exhaustive analysis of the PR which means that the newly added (whole) files will be included in the analysis"
    )
    parser.add_argument("--show-prompt",
                        action="store_true",
                        help="Shows the prompt which is the input for the AI (mainly used for debugging)")
    parser.add_argument("-x",
                        "--exclude-file-types",
                        nargs="+",
                        default=[],
                        help="Excluded file types from the analysis")
    parser.add_argument("-s",
                        "--service",
                        default="stash",
                        choices=["stash", "bitbucket", "github", "manual"],
                        help="Where do you have your PR?")

    # If there is a ~/.promethrc.json file, update the defualts so the user does not have to define them
    prometh_rc_filepath = Path("~/.promethrc.json").expanduser()
    if prometh_rc_filepath.exists():
        prometh_rc = json.load(prometh_rc_filepath.open())
        parser.set_defaults(**prometh_rc)

    args = parser.parse_args()

    if args.base_url is None:
        raise ValueError("Base URL is not set")
    if args.project_key is None:
        raise ValueError("Project key is not set")

    return args


class StashAPI:

    def __init__(self, base_url: str, project_key: str, repo_slug: str, pr_id: int) -> None:
        self.stash_token = os.environ["STASH_HTTP_ACCESS_TOKEN"]
        if self.stash_token is None:
            raise ValueError("STASH_HTTP_ACCESS_TOKEN is not set")
        self.base_url = base_url
        self.project_key = project_key
        self.repo_slug = repo_slug
        self.pr_id = pr_id

        self.headers = {"Authorization": f"Bearer {self.stash_token}", "Accept": "application/json"}
        # TODO: check if we can access the base URL (are we connected to the VPN?)

    def _get_repo_info_from_stash(self) -> dict:
        url_info = f"http://{self.base_url}/rest/api/latest/projects/{self.project_key}/repos/{self.repo_slug}/pull-requests/{self.pr_id}"
        response = requests.get(url_info, headers=self.headers)
        pr_info_data = response.json()
        return pr_info_data

    def _get_repo_diff_data_from_stash(self) -> dict:
        url_diff = f"http://{self.base_url}/rest/api/latest/projects/{self.project_key}/repos/{self.repo_slug}/pull-requests/{self.pr_id}/diff"
        response = requests.get(url_diff, headers=self.headers)
        pr_diff_data = response.json()
        return pr_diff_data

    def get_data(self) -> dict:
        pr_info_data = self._get_repo_info_from_stash()
        pr_diff_data = self._get_repo_diff_data_from_stash()

        # Last commit of the PR
        to_commit = pr_diff_data["toHash"]
        # Commit before the first commit of the PR
        from_commit = pr_diff_data["fromHash"]
        # Title of the PR
        pr_title = pr_info_data["title"]
        # Description of the PR (this is optional)
        try:
            pr_description = pr_info_data["description"]
        except KeyError:
            pr_description = ""
        # Link to the PR
        pr_link = pr_info_data["links"]["self"][0]["href"]

        return {
            "to_commit": to_commit,
            "from_commit": from_commit,
            "pr_title": pr_title,
            "pr_description": pr_description,
            "pr_link": pr_link,
        }


def generate_llm_response(gpt_type: str, messages: List[dict]) -> Tuple[str, Optional[dict]]:
    response: str = "No Response"
    usage_metrics: Optional[dict] = None

    if gpt_type in OPENAI_GPT_TYPES:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY is not set")

        chat_completion: dict = openai.ChatCompletion.create(model=gpt_type, messages=messages)
        response = chat_completion["choices"][0]["message"]["content"]
        usage_metrics = chat_completion["usage"]
    else:
        # If it's not an OpenAI GPT type, it's a custom locally deployed GPT type
        # TODO: make this endpoint configurable & the parameters
        url = "http://localhost:8080/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {"model": gpt_type, "messages": messages, "temperature": 0.7}
        resp = requests.post(url, headers=headers, data=json.dumps(data))
        response = resp.json()["choices"][0]["message"]["content"]
        usage_metrics = None

    return response, usage_metrics


def analyze_pr_with_GPT(pr_title, pr_description, diff_output, gpt_type, print_prompt: bool = False):
    prompt = f""" You are a senior engineer and your task is the following:
        - Review the code changes provided in a diff and provide feedback
        - Separately point out the bugs, security issues, missed best-practices
        - If there are any bugs, highlight them (and use 'BUG' tag at the start of the line).
        - Does the code do what it says in the pull request title, description?
        - Do not do nitpicking, discard minor issues.
        - Use markdown headers and bullet points to organize your feedback
        - In your output use the following headers: '**Summary**', '**Bugs**', '**Security**', '**Best-practices**', '**Other**'
        - Provide security recommendations if there are any.
        - Provide details on missed use of best-practices.
        - Be concise and to the point.

        You are provided with the code changes (diffs) in a unidiff format.
    """

    pr_description_message = f"""A description was given to help you assist in understand why these changes were made.
    The description was provided in a markdown format.

    Title: {pr_title}
    Description: {pr_description}
    """

    # TODO: Truncate the diff if it is possible and if the diff is too big
    # (the change does not need to be here, this is just a reminder)
    diff_message = f"""Diff in unidiff format:

    {diff_output}
    """

    final_message = """All code changes have been provided.
    Please provide me with your concise code review based on all the changes, context & title provided
    """

    messages = [{
        "role": "system",
        "content": prompt
    }, {
        "role": "user",
        "content": pr_description_message
    }, {
        "role": "user",
        "content": diff_message
    }, {
        "role": "user",
        "content": final_message
    }]

    if print_prompt:
        print("\n".join([message["content"] for message in messages]))

    response, usage_metrics = generate_llm_response(gpt_type, messages)

    # Add the analyzis message to the message for the interactive chatting
    messages.append({"role": "assistant", "content": response})

    return response, usage_metrics, messages


def main():
    console = Console()
    args = get_args()

    table = Table(title="Arguments")
    table.add_column("Argument", justify="right", style="bold", no_wrap=True)
    table.add_column("Value")
    for arg in vars(args):
        table.add_row(arg, str(getattr(args, arg)))
    console.print(table)

    # TODO: make it work with Bicbucket/Stash, GitHub, GitLab, w/out any of them (manual mode)
    api = None
    if args.service in {"stash", "bitbucket"}:
        api = StashAPI(args.base_url, args.project_key, args.repo, args.pull_request_id)
    elif args.service == "github":
        raise NotImplementedError("GitHub is not implemented yet")
    else:
        print("Manual mode...")
        raise NotImplementedError("Manual mode is not implemented yet")
    try:
        data: dict = api.get_data()
        to_commit = data["to_commit"]
        from_commit = data["from_commit"]
        pr_title = data["pr_title"]
        pr_description = data["pr_description"]
        pr_link = data["pr_link"]
    except Exception as e:
        print(f"Error: {e}")
        print("Problems... but I am not going to handle it. Happy debugging sucker")
        exit(1)

    table = Table(title="Info")
    table.add_column("Info", justify="right", style="bold", no_wrap=True)
    table.add_column("Value")
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(table)

    _cmd = ["git", "diff", "--name-status", from_commit, to_commit]
    diff_name_status_output = subprocess.check_output(_cmd, encoding="utf-8")

    _diff_filter = "M"
    if args.exhaustive_analysis:
        # Include all the lines from the added files as well
        _diff_filter += "A"

    _cmd = ["git", "diff", f"--diff-filter={_diff_filter}", f"-U{args.nb_context_lines}", from_commit, to_commit]
    if len(args.exclude_file_types) > 0:
        _cmd.append("--")
        _cmd.append(".")
        for t in args.exclude_file_types:
            _cmd.append(f":(exclude)*{t}")

    # print(f"git diff command:\n{' '.join(_cmd)}\n")

    diff_modified_output = subprocess.check_output(_cmd, encoding="utf-8")
    if len(diff_modified_output) == 0:
        print("No file modifications in the PR. But new and deleted files might be present.")

    diff_output = diff_name_status_output + "\n\n" + diff_modified_output

    if args.show_diff_only:
        console.print("[bold]Showing only the git diff output:[/bold]\n")
        console.print(diff_output)
        exit(0)

    if len(diff_output) == 0:
        raise Exception("No diff output. Something is wrong, but I don't care.")

    analyzis_message, usage_metrics, messages = analyze_pr_with_GPT(pr_title, pr_description, diff_output,
                                                                    args.llm_type, args.show_prompt)

    if usage_metrics:
        table = Table(title="OpenAPI Usage Metrics")
        table.add_column("Metric", justify="right", style="bold", no_wrap=True)
        table.add_column("Value")
        for key, value in usage_metrics.items():
            table.add_row(key, str(value))
        console.print(table)

    console.print("[magenta][bold]------ Analyzis: ------[/bold][/magenta]\n")
    analyzis_message_md = Markdown(analyzis_message)
    console.print(analyzis_message_md)
    print("\n")

    if args.interactive:
        console.print("[cyan][bold]------ Interactive chatting: ------[/bold][/cyan]\n")
        console.print("You can chat with the AI about the PR")
        console.print("Press `CTRL+C` or type 'exit' [red]to exit[/red] the interactive chatting\n")

        while True:
            user_input = console.input("[magenta][bold]You:[/bold][/magenta] ")
            if user_input == "exit":
                exit(0)
            print("\n")
            messages.append({"role": "user", "content": user_input})
            response, _ = generate_llm_response(args.llm_type, messages)
            messages.append({"role": "assistant", "content": response})
            console.print(f"[cyan][bold]AI:[/bold] {response}\n[cyan]")


if __name__ == "__main__":
    main()
