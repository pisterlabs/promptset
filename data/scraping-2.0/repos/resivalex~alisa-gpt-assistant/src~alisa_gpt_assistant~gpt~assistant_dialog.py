from .open_ai_client import OpenAiClient


class AssistantDialog:
    def __init__(self, client: OpenAiClient, assistant_id: str, thread_id: str):
        self.client = client
        self.assistant_id = assistant_id
        self.thread_id = thread_id

    def send(self, message: str) -> str:
        return self.client.ask_and_get_response(
            self.assistant_id,
            self.thread_id,
            message,
        )
