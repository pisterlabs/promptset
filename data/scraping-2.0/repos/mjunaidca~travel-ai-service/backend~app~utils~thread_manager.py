from openai.types.beta.thread import Thread
from openai import OpenAI


class CreateThread():
    def __init__(self, client: OpenAI):
        if client is None:
            raise Exception("OpenAI Client is not initialized")
        self.client = client
        print("CreateThread initialized with OpenAI client.")

    def create_thread(self, purpose: str = 'assistants') -> Thread:
        """Create a Thread."""
        print(f"Creating a new Thread for purpose: '{purpose}'...")
        thread: Thread = self.client.beta.threads.create()
        print(f"New Thread created. Thread ID: {thread.id}")
        return thread
