from alisa_gpt_assistant.protocols import DialogProtocol

from .open_ai_client import OpenAiClient
from .assistant_dialog import AssistantDialog


class AssistantDialogFactory:
    def __init__(self, open_api_key: str, assistant_id: str):
        self.client = OpenAiClient(open_api_key)
        self.assistant_id = assistant_id

    def create(self) -> DialogProtocol:
        thread = self.client.create_thread()

        return AssistantDialog(self.client, self.assistant_id, thread.id)
