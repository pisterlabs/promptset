import ast
import os

from legal_openai.openai_tasks import OpenaiTask


class RuleClassifier:
    def __init__(self, path):
        self.path = path
        self.prompt_path = os.environ.get('PROMPT_PATH')

    def openai_ruleclassifier(self, model_name='text-davinci-003', temperature=0,
                              top_p=1, api_key=None, prompt=None, article=None):
        if prompt is None:
            with open(self.prompt_path + '/rule_classifier_prompt.txt', 'r') as f:
                prompt = f.read()
        response = OpenaiTask(path=self.path, api_key=api_key).execute_task(
            article=article, prompt=prompt)
        return ast.literal_eval(response)
