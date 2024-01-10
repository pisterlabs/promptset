import logging
import os
import json
from dotenv import load_dotenv

import openai

from openai.embeddings_utils import get_embedding, cosine_similarity

from github import Github

load_dotenv()

from helpers.log_mod import logger


# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000

openai.api_key = os.getenv("OPENAI_API_KEY")


class SearchIssue:

    """
    Class SearchIssue:

    Parameters:
    df (DataFrame): A DataFrame of issues.

    Methods:
    generate_embeddings():
    Generates OpenAI embeddings for the combined issue title and description.

    Functionality:
    - Combines the issue title and description into a single column "combined"
    - Generates an embedding for each issue using the OpenAI text-embedding-ada-002 model
    - Adds an "embedding" column to the DataFrame with the embeddings
    - Returns the DataFrame

    find_similar_issues():
    Finds the n most similar issues to a new issue based on cosine similarity of the OpenAI embeddings.

    Parameters:
    new_issue (str): The text of the new issue.
    n (int): The number of similar issues to return.
    pprint (bool): Whether to print the first 200 characters of each result.

    Functionality:
    - Generates an embedding for the new_issue using the OpenAI text-embedding-ada-002 model
    - Calculates the cosine similarity between the new issue embedding and all issue embeddings
    - Adds a "similarity" column to the DataFrame with the cosine similarities
    - Filters to only keep issues with similarity &gt; 0.8
    - Sorts by similarity in descending order and takes the top n results
    - Replaces "Issue title: " and "; Issue description:" with empty strings
    - Optionally prints the first 200 characters of each result
    - Returns the top n similar issues
    """

    def __init__(self, df) -> None:
        self.df = df
        logger.info("SearchIssue object initialized.")

    def generate_embeddings(self):
        logger.info("Generating embeddings...")
        self.df["combined"] = (
            "Issue description: "
            + self.df.issue_description.str.strip()
            + "; Issue Title: "
            + self.df.issue_title.str.strip()
        )

        self.df["embedding"] = self.df.combined.apply(
            lambda x: get_embedding(x, engine=embedding_model)
        )

        logger.info("Embeddings generated.")
        return self.df

    def find_similar_issues(self, new_issue, n=3, pprint=True):
        logger.info(f"Finding {n} most similar issues to '{new_issue}'...")
        issue_embedding = get_embedding(new_issue, engine="text-embedding-ada-002")
        self.df["similarity"] = self.df.embedding.apply(
            lambda x: cosine_similarity(x, issue_embedding)
        )

        threshold = 0.85

        results = (
            self.df.sort_values("similarity", ascending=False)
            .query("similarity > @threshold")
            .head(n)
            .combined.str.replace("Issue title: ", "")
            .str.replace("; Issue description:", ": ")
        )

        similar_issues = []

        for r in results:
            issues_document = r[:50000]
            issues_body = issues_document.split(";")
            logger.info(f"Loggine issue body, {issues_body}")
            issue_description = issues_body[0]
            issues_title = issues_body[1]

            similar_issues.append(
                {
                    "issues_title": issues_title.split(":", 1)[1].strip(),
                    "issues_description": issue_description.split(":", 1)[1].strip(),
                }
            )

        logger.info(f"{len(similar_issues)} most similar issues found.")
        return similar_issues
