import os
import re

from legal_openai.openai_tasks import OpenaiTask


class DeonticLogic:
    def __init__(self, path):
        self.path = path
        self.prompt_path = os.environ.get('PROMPT_PATH')

    def openai_classifier(self, model_name='text-davinci-003', temperature=0,
                          top_p=1, api_key=None, prompt=None, article=None):
        """openai_classifier.

        Parameters
        ----------
        model_name :
            model_name for OpenAI
        temperature :
            temperature indicating the randomness 
        top_p :
            top_p
        api_key :
            api_key for OpenAI access
        prompt :
            prompt is the default template used for querying
        article :
            the article number of index id of the article to be used for querying
        """
        reponse_list = []
        if prompt is None:
            with open(self.prompt_path + '/deontic_logic/if_then.txt', 'r') as f:
                if_then_prompt = f.read()
            with open(self.prompt_path + '/deontic_logic/describe_if_then.txt', 'r') as f:
                describe_if_then_prompt = f.read()
            with open(self.prompt_path + '/deontic_logic/classify_deontic_logic.txt', 'r') as f:
                classify_deontic_logic_prompt = f.read()
            response = OpenaiTask(path=self.path, api_key=api_key).execute_task(
                article=article, prompt=if_then_prompt)
            for resp in response:
                final_response = {}
                if resp is not None:
                    final_response['if_then_rule'] = resp
                    prompt = re.sub("<IF_THEN_RULE>", resp, describe_if_then_prompt)
                    response = OpenaiTask(path=self.path, api_key=api_key).execute_task(
                        article=article, prompt=prompt)
                    final_response['conditions'] = response
                    prompt = re.sub("<IF_THEN_RULE>", resp, classify_deontic_logic_prompt)
                    response = OpenaiTask(path=self.path, api_key=api_key).execute_task(
                        article=article, prompt=prompt)
                    final_response['classification'] = response
                    reponse_list.append(final_response)
        return reponse_list

    def openai_if_then_fetch(self, model_name='text-davinci-003', temperature=0,
                             top_p=1, api_key=None, prompt=None, article=None, text=None):
        if prompt is None:
            with open(self.prompt_path + '/deontic_logic/if_then.txt', 'r') as f:
                if_then_prompt = f.read()
            if_then_prompt = re.sub("<IF_THEN_RULE>", text, if_then_prompt)
            response = OpenaiTask(path=self.path, api_key=api_key).execute_task(
                article=article, prompt=if_then_prompt)
            return response

    def openai_if_then_describe(self, model_name='text-davinci-003', temperature=0,
                                top_p=1, api_key=None, prompt=None, article=None, text=None):
        if prompt is None:
            with open(self.prompt_path + '/deontic_logic/describe_if_then.txt', 'r') as f:
                describe_if_then_prompt = f.read()
            describe_if_then_prompt = re.sub("<IF_THEN_RULE>", text, describe_if_then_prompt)
            response = OpenaiTask(path=self.path, api_key=api_key).execute_task(
                article=article, prompt=describe_if_then_prompt)
            return response

    def openai_classifier_logic_only(self, model_name='text-davinci-003', temperature=0,
                                     top_p=1, api_key=None, prompt=None, article=None, text=None):
        if prompt is None:
            with open(self.prompt_path + '/deontic_logic/classify_deontic_logic.txt', 'r') as f:
                classify_deontic_logic_prompt = f.read()
            classify_deontic_logic_prompt = re.sub("<IF_THEN_RULE>", text, classify_deontic_logic_prompt)
            response = OpenaiTask(path=self.path, api_key=api_key).execute_task(
                article=article, prompt=classify_deontic_logic_prompt)
            return response
