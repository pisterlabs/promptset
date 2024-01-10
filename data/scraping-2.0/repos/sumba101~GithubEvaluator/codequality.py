from openai_functions import response_gen
from prompts import REPO_MAP_CODE_QUALITY_PROMPT, README_READABILITY_PROMPT
from utils import fetch_map


class CodeQuality:
    def __init__(self, repo_name, readme):
        self.repomap = fetch_map(repo_name)
        self.readme = readme

    def _evaluate_code_quality(self):
        code_quality_report = response_gen(self.repomap, REPO_MAP_CODE_QUALITY_PROMPT, 0)
        readme_readability_report = response_gen(self.readme, README_READABILITY_PROMPT, 0)
        return code_quality_report, readme_readability_report
