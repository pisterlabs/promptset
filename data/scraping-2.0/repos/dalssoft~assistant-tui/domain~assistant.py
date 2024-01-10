from openai import OpenAI
from log import log_action


class Assistant:
    def __init__(self, id, assistant=None):
        self.client = OpenAI()
        self.id = id
        self.assistant = assistant
        self.name = assistant.name if assistant else None

    @staticmethod
    def create(name, instructions):
        # Create a new assistant
        client = OpenAI()
        my_assistant = client.beta.assistants.create(
            instructions=instructions,
            name=name,
            tools=[{"type": "code_interpreter"}],
            model="gpt-4-1106-preview",
        )
        assistant = Assistant.from_raw(my_assistant)
        log_action("Assistant", "create", assistant)
        return assistant

    def retrieve(self):
        my_assistant = self.client.beta.assistants.retrieve(self.id)
        self.assistant = my_assistant
        log_action(self, "retrieve", my_assistant)
        return self

    def delete(self):
        response = self.client.beta.assistants.delete(self.id)
        log_action(self, "delete", response)

    @staticmethod
    def list_all():
        client = OpenAI()

        my_assistants = client.beta.assistants.list(
            order="desc",
            limit=100,
        )

        # transform raw assistants into Assistant objects
        assistants = []
        for assistant in my_assistants.data:
            assistant = Assistant.from_raw(assistant)
            assistants.append(assistant)

        log_action("Assistant", "list_all", assistants)

        return assistants

    @staticmethod
    def from_raw(assistant):
        return Assistant(assistant.id, assistant)
