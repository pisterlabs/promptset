import pandas as pd
import os
import logging

from langchain.document_loaders import GitHubIssuesLoader
from dotenv import load_dotenv


GITHUB_REPOSITORY = "langchain-ai/langchain"
GITHUB_LABEL = "langchain"
STORE_PATH = "data-pipeline"


def load_environment_vars() -> dict:
    """Load required environment variables. Raise an exception if any are missing."""

    load_dotenv()
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

    if not github_token:
        raise EnvironmentError(
            "GITHUB_PERSONAL_ACCESS_TOKEN environment variable not set.")

    logging.info("Environment variables loaded.")
    return {"GITHUB_TOKEN": github_token}


def initialize_github_loader(repo: str) -> GitHubIssuesLoader:
    """Init GitHubIssuesLoader."""

    logging.info(f"Initializing GitHubIssuesLoader for repo: {repo}")
    loader = GitHubIssuesLoader(
        repo=repo,
        include_prs=False,
    )
    return loader


def fetch_as_df(loader: GitHubIssuesLoader) -> pd.DataFrame:
    """Fetch Data from GitHub Repository"""

    logging.info(f"Fetching 'issues' from Github: '{loader.repo}'")
    docs = loader.load()

    df = pd.DataFrame()
    for doc in docs:
        description = doc.page_content
        metadata = doc.metadata

        row = pd.DataFrame([{'description': description, **metadata}])
        df = pd.concat([df, row], ignore_index=True)

    logging.info(f"Fetched {len(df)} 'issues' from Github: '{loader.repo}'")
    logging.info(f"Dataframe Columns: {df.columns}")

    return df


def store_as_pickle(df: pd.DataFrame, label: str, path: str):
    """Store DataFrame as json to local file system"""
    file_name = f"{label}-github-issues-{pd.Timestamp.today().strftime('%Y-%m-%d')}.pkl"

    logging.info(f"Storing 'issues' to '{path}/{file_name}'")
    df.to_pickle(f"{path}/{file_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        env_vars = load_environment_vars()

        loader = initialize_github_loader(GITHUB_REPOSITORY)
        df = fetch_as_df(loader)
        store_as_pickle(df, label=GITHUB_LABEL, path=STORE_PATH)
    except EnvironmentError as ee:
        logging.error(f"Environment Error: {ee}")
        raise
    except Exception as ex:
        logging.error(f"Unexpected Error: {ex}")
        raise
