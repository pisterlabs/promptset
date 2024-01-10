import os

import click
import git
import openai
from common_bin_utils import parse_month

from common_utils.logger import logger


def generate_git_log(repo_name, after, before):
    logger.info(f"Generating git log for {repo_name} between {after} and {before}")
    repo = git.Repo("../deep-learning" if repo_name == "deep-learning" else ".")
    repo.git.fetch()
    log = repo.git.log(
        '--pretty=format:"%s%n%n%b"',
        f'--after="{after}"',
        f'--before="{before}"',
        "origin/main" if repo_name == "deep-learning" else "origin/develop",
    )
    return log


def get_chatgpt_summary(git_log: str, summary_type: str = "non_technical") -> str:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if summary_type == "non_technical":
        prompt = (
            f"Summarize the following git log avoiding too technical words. "
            f"This is to be presented to investors/board members, non technical people. "
            f"Provide 2 to 5 bullet points maximum:\n{git_log}"
        )
    else:
        prompt = (
            f"Summarize the following git log in 2 to 5 technical points:\n{git_log}"
        )
    logger.info(f"Querying OpenAI with prompt with length {len(prompt)}")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        n=1,
        temperature=0.4,
    )

    summary = response.choices[0]["message"]["content"].strip()
    return summary


@click.command()
@click.option(
    "--month",
    default=None,
    help='The month to generate the summary for in the format "YYYY-MM" or a month name like "March". '
    "Defaults to current month.",
)
@click.option(
    "--mode",
    default="investors",
    help="The mode to run the report in. Can be 'investors' or 'technical'.",
)
@click.option(
    "--repo",
    default="slam",
    help="The repo to generate the summary for. Can be 'slam' or 'deep-learning'. "
    "Assumes deep-learning is located in common parent folder",
)
def main(month: str, mode: str, repo: str):
    start_date, end_date = parse_month(month)

    git_log = generate_git_log(repo_name=repo, after=start_date, before=end_date)

    summary_type = "non_technical" if mode == "investors" else "technical"

    summary = get_chatgpt_summary(git_log, summary_type=summary_type)

    logger.info(f"{summary_type} summary for {month} for repo {repo}:")
    logger.info(summary)


if __name__ == "__main__":
    main()
