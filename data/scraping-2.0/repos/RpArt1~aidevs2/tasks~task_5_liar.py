import requests
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage

from tasks.abstractTask import AbstractTask
from utils.assigment_utils import AssigmentUtils


class LiarTask(AbstractTask):
    TASK_NAME = 'liar'
    url = "https://zadania.aidevs.pl/task/"

    def solve_task(self):
        super().solve_task()

    def process_task_details(self):

        my_question = "What is capital of Poland?"
        # pobierz zadanie z aidevs wysyłając pytanie

        data = {
            "question": my_question
        }
        key = AssigmentUtils.get_key_for_assigment(self.TASK_NAME)
        url = self.url + key
        self.url = url
        self.key = key
        response = requests.post(self.url, data=data)
        answer = None
        if response.status_code == 200:
            print("Request was successful!")
            json_data = response.json()
            answer = json_data.get('answer')
            print(answer)
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response content:", response.text)

        # GUARDRAILS

        chat = ChatOpenAI()
        guard_prompt = 'Return ONLY 1 or 0 and nothing else if the response: ' + answer + 'to question:' + my_question + " is correct "
        response = chat.invoke(guard_prompt).content
        print(f"Guardrails response: [{response}]")
        if int(response) == 1:
            return "YES"
        return "NO"
