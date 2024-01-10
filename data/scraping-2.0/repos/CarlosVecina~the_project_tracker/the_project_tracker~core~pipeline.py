import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from the_project_tracker.core.data_models import PR, Release
from the_project_tracker.core.explainer import OpenAIExplainer
from the_project_tracker.core.github_retriever import (GitHubRetrieverPRs,
                                                       GitHubRetrieverReleases)
from the_project_tracker.core.utils import parse_github_url
from the_project_tracker.db.pg_conn import PGDataConnection, SettingsSSH
from the_project_tracker.db.sqlite_conn import SQLiteDataConnection

load_dotenv()


class ReleasePipeline(BaseModel):
    class Config:
        title = "Pipeline Release Tracker"
        arbitrary_types_allowed = True

    repo_url: str
    max_releases_num: int = Field(5)
    connection: PGDataConnection | SQLiteDataConnection = (
        PGDataConnection(ssh_config=SettingsSSH())
    )  # SQLiteDataConnection(db_name="tracker_db.sqlite")

    def run(self):
        owner, repo = parse_github_url(self.repo_url)
        gh = GitHubRetrieverReleases(owner=owner, repo=repo)
        list_last_releases = gh.get_last_release(max_releases_num=self.max_releases_num)
        for c_rel in list_last_releases:
            db = self.connection
            rows = db.query_database(
                f"SELECT repo_url, tag_name FROM {db.releases_table} WHERE repo_url = '{self.repo_url}' and tag_name =  '{c_rel['tag_name']}'",
            )
            if len(rows) >= 1:
                print(f"Release {c_rel['tag_name']} already tracked in this project")
                continue
            else:
                explainer = OpenAIExplainer()
                # Retrieve code diffs if needed
                code_diffs = None

                explanation = explainer.explain(
                    repo=repo,
                    title=c_rel["tag_name"],
                    body=c_rel["body"],
                    entity="release",
                    code_diffs=code_diffs,
                )
                explanation_es = explainer.explain_es(
                    repo=repo,
                    title=c_rel["tag_name"],
                    body=c_rel["body"],
                    entity="release",
                    code_diffs=code_diffs,
                )
                print(
                    f"\nPR {c_rel['tag_name']}. Project: {repo} Explanation:{explanation}\n"
                )

                release_obj = Release(
                    repo_url=self.repo_url,
                    name=c_rel["name"] or '',
                    tag_name=c_rel["tag_name"],
                    published_at=c_rel["published_at"],
                    assets=c_rel["assets"],
                    body=c_rel["body"],
                    explanation=explanation,
                    explanation_es=explanation_es,
                    inserted_at=str(datetime.datetime.now()),
                    updated_at=str(datetime.datetime.now()),
                )

                db.insert_release(release_obj)


class PullRequestPipeline(BaseModel):
    class ReleasePipeline(BaseModel):
        class Config:
            title = "Pipeline PRs Tracker"

        arbitrary_types_allowed = True

    repo_url: str
    max_pr_num: int = Field(5)
    include_code_diffs: bool = True
    connection: PGDataConnection | SQLiteDataConnection = PGDataConnection()

    def run(self, fail_if_new=False):
        owner, repo = parse_github_url(self.repo_url)
        gh = GitHubRetrieverPRs(owner=owner, repo=repo)
        list_last_prs = gh.retrieve(max_pr_num=self.max_pr_num)

        if len(list_last_prs) == 0:
            print("No merged PRs found in the repository.")
            return

        for c_pr in list_last_prs:
            # Check if the PR is tracked
            db = self.connection
            rows = db.query_database(
                f"SELECT repo_url, pr_id FROM {db.prs_table} WHERE repo_url = '{self.repo_url}' and pr_id = {c_pr['number']} and inc_code_diffs = { int(self.include_code_diffs)}",
            )

            if (fail_if_new) | (len(rows) >= 1):
                print(f"PR {c_pr['number']} already tracked in this project")
                continue
            else:
                explainer = OpenAIExplainer()
                # Retrieve code diffs if needed
                code_diffs = None
                if self.include_code_diffs:
                    code_diffs = gh.get_code_diffs(c_pr["number"])

                explanation = explainer.explain(
                    repo=repo,
                    title=c_pr["title"],
                    body=c_pr["body"],
                    code_diffs=code_diffs,
                )
                explanation_es = explainer.explain_es(
                    repo=repo,
                    title=c_pr["title"],
                    body=c_pr["body"],
                    code_diffs=code_diffs,
                )
                print(
                    f"\nPR {c_pr['number']}. Project: {repo} Explanation:{explanation}\n"
                )
                pr_obj = PR(
                    repo_url=self.repo_url,
                    pr_id=c_pr["number"],
                    inc_code_diffs=int(self.include_code_diffs),
                    merged_at=str(c_pr["merged_at"]),
                    inserted_at=str(datetime.datetime.now()),
                    updated_at=str(datetime.datetime.now()),
                    pr_title=c_pr["title"],
                    pr_body=c_pr["body"],
                    commits_url=c_pr["commits_url"],
                    explanation=explanation,
                    explanation_es=explanation_es
                )

                db.insert_pr(pr_obj)
