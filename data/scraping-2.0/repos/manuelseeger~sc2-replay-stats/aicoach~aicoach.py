from openai import OpenAI
from openai.types.beta import Thread, Assistant
from openai.types.beta.threads import Run
from dotenv import load_dotenv
import os
import time
import json
from .functions import AIFunctions
from config import config
import logging
from typing import Dict

log = logging.getLogger(config.name)
log.setLevel(logging.INFO)

load_dotenv()


ASSISTANT_ID = os.environ["ASSISTANT_ID"]

client = OpenAI()


def wait_on_run(run, thread, statuses=[]):
    while (
        run.status == "queued" or run.status == "in_progress" or run.status in statuses
    ):
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.3)
    return run


class AICoach:
    current_thread_id: str = None
    threads: Dict[str, Thread] = {}
    thread: Thread = None

    def __init__(self):
        self.assistant: Assistant = client.beta.assistants.retrieve(
            assistant_id=ASSISTANT_ID
        )
        self.functions = {f.__name__: f for f in AIFunctions}

    def get_most_recent_message(self):
        messages = client.beta.threads.messages.list(thread_id=self.thread.id)
        if messages.data[0].role != "assistant":
            log.warn("Assistant sent no message")
            return ""
        return messages.data[0].content[0].text.value

    def create_thread(self, message=None):
        self.thread: Thread = client.beta.threads.create()
        self.threads[self.thread.id] = self.thread

        self.current_thread_id = self.thread.id

        if message:
            client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=message,
            )

    def create_run(self) -> Run:
        run = client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
        )
        run = wait_on_run(run, self.thread)
        return run

    def chat(self, text) -> str:
        message = client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=text,
        )

        self.evaluate_run()
        # run_steps = client.beta.threads.runs.steps.list(
        #    thread_id=self.thread.id, run_id=self.run.id
        # )
        return self.get_most_recent_message()

    def evaluate_run(self, run=None) -> Run:
        if not run:
            run = self.create_run()
        run = wait_on_run(run, self.thread)
        if run.status == "requires_action":
            outputs = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                if tool_call.type == "function":
                    args = json.loads(tool_call.function.arguments)
                    name = tool_call.function.name
                    output = self.call_function(run, tool_call.id, name, args)
                    outputs.append(output)

            client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread.id,
                run_id=run.id,
                tool_outputs=outputs,
            )
            run = wait_on_run(run, self.thread, statuses=["requires_action"])
        # run = self.evaluate_run(run)
        if run.status == "completed":
            return run

    def call_function(self, run, tool_call_id, name, args) -> Run:
        log.debug(name, args)
        log.info('Calling function "{}" with args: {}'.format(name, args))
        result = self.functions[name](**args)
        # log.debug(result)

        output = {
            "tool_call_id": tool_call_id,
            "output": json.dumps(result, default=str),
        }
        return output


def main():
    pass


if __name__ == "__main__":
    main()
