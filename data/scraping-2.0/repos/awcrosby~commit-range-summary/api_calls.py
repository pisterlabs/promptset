import os
from typing import Any, Dict, Generator, Optional

import httpx
from dotenv import load_dotenv
from jsonschema import validate

load_dotenv()
GITHUB_API_KEY = os.environ.get("GITHUB_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

COMMIT_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "stats": {
            "type": "object",
            "properties": {
                "additions": {"type": "number"},
                "deletions": {"type": "number"},
                "total": {"type": "number"},
            },
            "additionalProperties": False,
        },
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "status": {"type": "string"},
                    "changes": {"type": "number"},
                    "additions": {"type": "number"},
                    "deletions": {"type": "number"},
                    "patch": {"type": "string"},
                },
                "additionalProperties": False,
                "required": ["filename"],
            },
        },
    },
    "additionalProperties": False,
    "required": ["message", "files"],
}


class GitHubApiClient:
    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {GITHUB_API_KEY}",
        }

    def _make_get_request(self, url: str, params: dict[str, str]) -> httpx.Response:
        with httpx.Client() as client:
            r = client.get(url, headers=self.headers, params=params)

        if r.status_code != 200:
            message = r.json().get("message", "")
            raise RuntimeError(f"Error calling github api {url}, {r.status_code}: {message}")

        return r

    def get_commit(self, commit_sha: str) -> Dict[str, Any]:
        commit_blob = self._get_commit_blob(commit_sha)
        return self._transform_to_commit_schema(commit_blob)

    def get_commits(
        self, since: str, until: str, author: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        shas = self._get_shas(since, until, author)
        return [self.get_commit(sha) for sha in shas]

    def _get_shas(
        self, since: str, until: str, author: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Get list of commit shas for a date range."""
        gh_commit_list_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/commits"
        params = {"since": since, "until": until}
        if author:
            params["author"] = author

        resp = self._make_get_request(gh_commit_list_url, params)
        yield from (commit["sha"] for commit in resp.json())

        while "next" in resp.links:
            resp = self._make_get_request(
                resp.links["next"]["url"],
                params,
            )
            yield from (commit["sha"] for commit in resp.json())

    def _get_commit_blob(self, commit_sha: str) -> Dict[str, Any]:
        gh_commit_url = (
            f"https://api.github.com/repos/{self.owner}/{self.repo}/commits/{commit_sha}"
        )
        return self._make_get_request(gh_commit_url, params={}).json()

    def _transform_to_commit_schema(self, commit_blob: Dict[str, Any]) -> Dict[str, Any]:
        """Transform commit blob to match schema."""
        FILE_KEYS = ["patch", "filename", "status", "additions", "deletions", "changes"]
        files = []
        for file in commit_blob["files"]:
            d = {k: v for k, v in file.items() if k in FILE_KEYS}
            files.append(d)

        transformed_commit = {
            "message": commit_blob["commit"]["message"],
            "stats": commit_blob["stats"],
            "files": files,
        }

        validate(transformed_commit, schema=COMMIT_SCHEMA)
        return transformed_commit


class OpenAIApiClient:
    def __init__(self):
        self.openai_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }

    def _make_post_request(self, data: Dict[str, Any]) -> httpx.Response:
        with httpx.Client() as client:
            r = client.post(self.openai_url, headers=self.headers, json=data, timeout=90.0)

        if r.status_code != 200:
            try:
                message = r.json()["error"]["message"]
            except KeyError:
                message = ""
            raise RuntimeError(f"Error calling openai api, {r.status_code}: {message}")

        return r

    def generate_chat_completion(self, content: str) -> str:
        """Call OpenAI API with content and return the AI response."""
        # MODEL = "gpt-4-1106-preview"
        # MODEL_INPUT_CONTEXT_WINDOW_TOKENS = 128000
        # MODEL_TPM_LIMIT = 150000

        MODEL = "gpt-3.5-turbo-1106"
        MODEL_INPUT_CONTEXT_WINDOW_TOKENS = 16385
        MODEL_TPM_LIMIT = 90000

        CHAR_PER_TOKEN = 3.9  # usually 4, can reduce to be less likely to hit limit
        token_limit = min(MODEL_TPM_LIMIT, MODEL_INPUT_CONTEXT_WINDOW_TOKENS)
        token_estimate = int(len(content) / CHAR_PER_TOKEN)
        if token_estimate > token_limit:
            raise RuntimeError(
                f"Token estimate {token_estimate} exceeds maximum {token_limit} tokens for OpenAI API"
            )
        print(f"token_estimate: {token_estimate}")

        data = {"model": MODEL, "messages": [{"role": "user", "content": content}]}
        r = self._make_post_request(data)

        try:
            ai_reply = r.json()["choices"][0]["message"]["content"]
        except KeyError:
            raise RuntimeError("Error parsing response from OpenAI API")
        return ai_reply
