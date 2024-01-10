"""
Handler for issues
"""
import os

import github.PaginatedList
import pinecone
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from PyInquirer import prompt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from . import utilities
from .gitpy import GitPy
from .llms import OpenAI
from .questions import Questions


class IssueManager:
    """
    Issue manager class
    """

    CHOICES = {"All issues": "all", "Open issues": "open", "Closed issues": "closed"}

    def __init__(self, logger):
        """
        Initialize the issue manager

        openai_agent: OpenAI agent
        git_helper: GitPy object

        """
        self.openai_agent = OpenAI(
            os.getenv("OPENAI_API_KEY"),
            embeddings_model="text-embedding-ada-002",
        )
        self.logger = logger

        self.git_helper = GitPy(os.getenv("GITHUB_TOKEN"), logger=logger)
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east1-gcp")
        self.pinecone_index = pinecone.Index("gitbrew")
        self.actions = {
            "List Issues": self._list_issues,
            "Create an Issue": self._create_issue,
            "Find Duplicate Issues": self._find_duplicate_issues,
            "Find Similar Issues": self._find_similar_issues,
            "Cancel": self.exit,
        }
        self.issue_template = "Title: {title}\n Body: {body}"

    def __del__(self):
        pass

    def exit(self, arg):
        """
        Handler for "exit" keyword
        :param arg: Optional args
        :return: True
        """
        self.logger.info("Exiting issue manager")
        return True

    def handle(self):
        """
        Handle issues
        Loops until user chooses to cancel/exit

        repo options: remote or custom

        Provide these functionalities
        1. List issues
        2. Create issues
        3. Find duplicate issues in a repo
        4. Find similar issues before raising a new one
        :return: None
        """
        self.git_helper.repo_name = (
            utilities.get_repo_url()
        )  # will be valid or throw an error
        self.git_helper.set_repo()
        self.logger.info(f"Using repo: {self.git_helper.repo_name}")
        user_intention = prompt(Questions.ISSUE_INTERACTION_QUESTIONS)["choice"]
        self.logger.info(f"User intention: {user_intention}")
        while user_intention != "Cancel":
            self.actions.get(user_intention, self.exit)()
            user_intention = prompt(Questions.ISSUE_INTERACTION_QUESTIONS)["choice"]

    def _list_issues(self):
        """
        List issues. Default state is open.

        :return: None
        """
        questions = Questions.ISSUE_STATUS_QUESTIONS
        state = self.CHOICES.get(prompt(questions)["choice"])
        self.logger.info(f"Listing issues with state: {state}")
        self.git_helper.list_issues(state)

    def _create_issue(self):
        """
        Create an issue in the repository
        Prompts for title and body and
        creates an issue using the gitpy module
        without any further user interaction

        :return: None
        """
        title, body = self._get_issue_description()
        self.logger.info(f"Creating issue with title: {title}")
        self.git_helper.create_issue(title, body)

    def _find_duplicate_issues(self, threshold=0.8):
        """
        Group duplicate issues in the repository

        :return: None
        """

        all_issues = self.git_helper.fetch_issues(state="open")
        all_embeddings = {
            issue.number: self.openai_agent.create_embedding(
                self.issue_template.format(title=issue.title, body=issue.body)
            )["data"][0]["embedding"]
            for issue in tqdm(all_issues)
        }
        groups = self._generate_similarity_groups(all_embeddings, threshold)
        utilities.print_dictionary(groups, headers=["Title", "URL", "Similarity"])

    def _generate_similarity_groups(self, all_embeddings, threshold):
        """
        Generate similarity groups for duplicate issues
        :param all_embeddings: mapping of issue id to embeddings
        :param threshold: threshold for cosine similarity
        :return:
        """
        groups = {}
        items = all_embeddings.items()
        for issue_id, embed in tqdm(items):
            similar_issues = []
            for other_issue_id, other_embed in items:
                if issue_id == other_issue_id:
                    continue
                similarity = cosine_similarity([embed], [other_embed])[0][0]
                if similarity > threshold:
                    issue = self.git_helper.get_issue(other_issue_id)
                    similar_issues.append((issue.title, issue.html_url, similarity))
            similar_issues.sort(key=lambda x: x[2], reverse=True)
            groups[self.git_helper.get_issue(int(issue_id)).title] = similar_issues
        return groups

    def _find_similar_issues(self, n=10):
        """
        Finds similar issues that have been raised in the repository.
        Uses OpenAI's semantic search to find similar issues
        to avoid duplication.
        To be used before creating a new issue

        :return: None
        """
        new_issue_embedding = self.get_new_issue_embedding()
        issues: github.PaginatedList.PaginatedList = self.git_helper.fetch_issues(
            state="open"
        )
        if not issues.totalCount:
            print("--- There are no open issues in this repository. ---")
            self.logger.info("No issues found. Exiting...")
            return
        last_opened_issue_number = max((issue.number for issue in issues))
        matches = self.generate_matches_from_pinecone(
            last_opened_issue_number, n, new_issue_embedding, issues
        )
        repo = self.git_helper.github.get_repo(self.git_helper.repo_name)

        data = [
            (
                repo.get_issue(int(id_["id"])).title,
                repo.get_issue(int(id_["id"])).html_url,
            )
            for id_ in tqdm(sorted(matches, key=lambda x: x["score"], reverse=True))
            if id_["id"] != "LAST"
        ]
        utilities.print_table(data, headers=["Title", "URL"], show_index=True)

    def generate_matches_from_pinecone(
        self, last_opened_issue_number, n, new_issue_embedding, issues
    ):
        """
        Generate matches from pinecone db
        If the most recent issue in the repository is in pinecone db, fetch directly form pinecone db
        otherwise upsert all issues and their embeddings
        :param last_opened_issue_number:
        :param n:
        :param new_issue_embedding:
        :param issues:
        :return:
        """
        values = self.query_db([last_opened_issue_number] * 1536, 1, True)
        db_last_opened_issue_number = values[0]["values"][0] if values else None
        self.logger.info(
            f"Last opened issue number in the repo: {last_opened_issue_number}"
        )
        self.logger.info(
            f"Last opened issue number in the db: {db_last_opened_issue_number}"
        )
        if last_opened_issue_number == db_last_opened_issue_number:
            self.logger.info("All issues are already pineconed. Skipping upsert.")
        else:
            self.logger.info("Upserting embeddings into pinecone db.")
            vectors = self.create_vectors(issues)
            self.upsert_embeddings(
                vectors, self.git_helper.repo_name, last_opened_issue_number
            )
        matches: list = self.query_db(embedding=new_issue_embedding, n=n)
        return matches

    def get_new_issue_embedding(self):
        """
        Prompts the user for the issue title and description
        Returns the embeddings for the issue text

        :return: Embeddings for the issue text
        """
        title, body = self._get_issue_description()
        issue_text = self.issue_template.format(title=title, body=body)
        issue_text = simple_preprocess(remove_stopwords(issue_text), deacc=True)
        return self.openai_agent.create_embedding(issue_text)["data"][0]["embedding"]

    @staticmethod
    def _get_issue_description():
        """
        Get the issue title and description from the user
        :return: title, body
        """
        title = input("What's the title of the new issue? ")
        body = input("What's the description the new issue? ")
        return title, body

    def query_db(self, embedding, n, return_values=False):
        """
        Query the database for similar issues
        and return the top n matches

        :param return_values:
        :param embedding: new issue embedding
        :param n: number of similar issues to return
        :return: list of "matches" dicts with id, score, values etc
        """
        try:
            response = self.pinecone_index.query(
                namespace=self.git_helper.repo_name,
                top_k=n,
                vector=embedding,
                include_values=return_values,
            )
            return response["matches"]
        except pinecone.exceptions.PineconeException as e:
            self.logger.error(
                f"Something went wrong while querying items in pinecone db.: {e}"
            )

    def upsert_embeddings(self, vectors, namespace, last_opened_issue_number):
        """
        Upsert the embedding for an issue into the database

        :param last_opened_issue_number:
        :param vectors: list of vectors
        :param namespace: namespace (repository name)

        :return: None
        """
        try:
            self.pinecone_index.upsert(
                vectors=vectors,
                namespace=namespace,
            )

            self.pinecone_index.upsert(
                vectors=[("LAST", [last_opened_issue_number] * 1536)],
                namespace=namespace,
            )
        except pinecone.exceptions.PineconeException as e:
            self.logger.error(
                f"Something went wrong while updating items in pinecone db.: {e}"
            )

    def create_vectors(self, issues):
        """
        Create vectors for a list of issues
        Returns a list of tuples -> (issue_number, embeddings for the issue text)

        :param issues: list of issues
        :return: list of tuples with issue number and it's embeddings
        """
        return [
            (str(issue.number), self._generate_embeddings(issue))
            for issue in tqdm(issues)
        ]

    def _generate_embeddings(self, issue):
        """
        Returns vector embeddings for the issue
        :param issue: GitHub Issue object
        :return: Embeddings for the issue text
        """
        issue_text = self.issue_template.format(title=issue.title, body=issue.body)
        issue_text = simple_preprocess(remove_stopwords(issue_text), deacc=True)
        return self.openai_agent.create_embedding(issue_text)["data"][0]["embedding"]
