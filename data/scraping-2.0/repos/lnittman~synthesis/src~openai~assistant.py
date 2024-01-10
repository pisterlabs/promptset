#!/usr/bin/env python3
from openai import OpenAI

class Assistant:
    ASSISTANT_ID = "asst_Qm68Qvw3qCUFNluVnkUcxjVk"

    def __init__(self, key):
        self.openai_client = OpenAI()

    def commit(self, repo, sha):
        pass

    def branch(self, repo, tag):
        pass

    def pull_request(self, repo, number):
        pass

    def repo(self, repo):
        pass

    def project(self, repos):
        pass

