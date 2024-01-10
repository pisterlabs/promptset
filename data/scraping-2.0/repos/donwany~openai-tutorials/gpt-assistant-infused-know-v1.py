from typing import Literal

from openai import OpenAI
import time
from dotenv import load_dotenv
import os

load_dotenv()


class AssistantManager:
    def __init__(self, api_key, model="gpt-4-1106-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.file_id = None
        self.assistant = None
        self.thread = None
        self.run = None

    def create_file(self, file_path, purpose='assistants'):
        with open(file_path, "rb") as file:
            file_obj = self.client.files.create(file=file, purpose=purpose)
            self.file_id = file_obj.id

    def create_assistant(self, name, instructions, tools):
        self.assistant = self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=self.model,
            tools=tools,
            file_ids=[self.file_id]
        )

    def create_thread(self):
        self.thread = self.client.beta.threads.create()

    def add_message_to_thread(self, role, content):
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role=role,
            content=content
        )

    def run_assistant(self, instructions):
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=instructions
        )

    def wait_for_completion(self):
        while True:
            time.sleep(5)
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id
            )
            print(run_status.model_dump_json(indent=4))

            if run_status.status == 'completed':
                self.process_messages()
                break
            else:
                print("Waiting for the Assistant to process...")

    def process_messages(self):
        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)

        for msg in messages.data:
            role = msg.role
            content = msg.content[0].text.value
            print(f"{role.capitalize()}: {content}")


def main():
    api_key = os.getenv("api_key")
    manager = AssistantManager(api_key)

    manager.create_file("songs.csv")
    manager.create_assistant(
        name="Data Analyst Assistant",
        instructions="You are a personal Data Analyst Assistant",
        tools=[{"type": "code_interpreter"}]
    )
    manager.create_thread()
    manager.add_message_to_thread(role="user", content="What are the most popular tracks and their artists?")
    manager.run_assistant(instructions="What are the most popular tracks and their artists?")
    manager.wait_for_completion()


if __name__ == '__main__':
    main()
