import openai
import requests
import sys
from django.conf import settings
from suggestions.models import Snippet


def get_role(snippet_type):
    dict_role = {'FRAMING': 'system',
                 'ASSISTANT MESSAGE': 'assistant',
                 'USER MESSAGE': 'user',
                 'LLM MESSAGE': 'assistant'}
    return dict_role[snippet_type]


class Suggestion:
    # Suggest from Convo ID
    # messages [{"role": role, "content": content}, {"role": role, "content": content}]
    # role:snippet_type, content:text

    def __init__(self, convo, query):
        self.convo = convo
        self.query = query

    def build_message(self):
        snippets = Snippet.objects.filter(convo=self.convo, snippet_type="FRAMING")
        messages = [{"role": get_role(snippet.snippet_type), "content": snippet.text} for snippet in snippets]
        messages.append({"role": "user", "content": self.query})
        return messages

    def process_llm_call(self):
        openai.api_key = settings.AZURE_OPENAI_KEY
        openai.api_base = settings.AZURE_OPENAI_ENDPOINT
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15'  # put in .env?
        deployment_name = settings.GPT35_DEPLOY_NAME
        messages = self.build_message()
        import time
        start = time.time()
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=messages,
            n=1
        )
        end = time.time()
        print("DIAGNOSE OPEN AI CALL", end-start)
        print(response)
        # llm_response = "1. Captain America (movie) \n 2. Hulk (movie) \n 3. Avengers (movie)"
        llm_response = response["choices"][0]["message"]["content"]

        return llm_response

    def process_llm_response(self):
        # # Todo save the tokens so we can calculate how much the cost for every convo
        resp_json = {"items": []}
        try:
            BASE_LLM_URL = settings.BASE_LLM_URL
            PROCESS_LLM_URL = settings.PROCESS_LLM_URL
            URL = BASE_LLM_URL + PROCESS_LLM_URL
            llm_response = self.process_llm_call()

            import time
            start = time.time()
            resp = requests.post(URL, json={"llm_response": llm_response})
            end = time.time()
            print("DIAGNOSE OPEN AI FROM MIKE", end-start)
            if resp.status_code != 200:
                print(resp.status_code)
                return resp_json
        except:
            e = sys.exc_info()
            print(e)
            return resp_json
        resp_json = resp.json()
        return resp_json
