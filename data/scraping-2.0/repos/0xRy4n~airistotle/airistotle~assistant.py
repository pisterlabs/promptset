# Built-ins
import time
import json

# Third-party
import backoff
import openai

from openai import BadRequestError

# AIRISTOTLE
from .logger import GlobalLogger
from .settings import AVAILABLE_PLUGINS


class Assistant:
    def __init__(self, openai_api_key: str, assistant_id: str, thread_id: str = ""):
        self.client = openai.Client(api_key=openai_api_key)
        self.assistant = self.client.beta.assistants.retrieve(assistant_id=assistant_id)
        self.log = GlobalLogger("Assistant")

        # If a thread_id is provided, use it, otherwise create a new thread
        if thread_id:
            self.thread_id = thread_id
            self.log.debug(f"Using existing thread: {self.thread_id}")
        else:
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id
            self.log.info(f"Created new thread: {self.thread_id}")

    def process_requires_action(self, run):
        tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
        func = tool_call.function
        params = json.loads(func.arguments)

        self.log.debug(f"Processing function call on '{func.name}'")
        self.log.audit(f"Function call params: {params}")

        if func.name in AVAILABLE_PLUGINS:
            plugin = AVAILABLE_PLUGINS[func.name]
            result = plugin.run(**params)
        else:
            result = f"An error occurred: function '{func.name}' could not be found."
            self.log.warning(f"Function call on '{func.name}' not found.")

        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread_id,
            run_id=run.id,
            tool_outputs=[{"tool_call_id": tool_call.id, "output": result}],
        )

    def get_response(self):
        self.log.debug("Getting latest assistant message.")
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id).data
        assistant_messages = [
            msg for msg in messages if msg.role in ["assistant", "system"]
        ]
        self.log.audit(f"Got latest assistant message: {assistant_messages[0]}")
        return (
            assistant_messages[0].content[0].text.value if assistant_messages else None  # type: ignore
        )

    @backoff.on_exception(backoff.expo, BadRequestError, max_time=60)
    def send_message(self, user_input):
        self.log.debug("Sending message to assistant.")
        self.client.beta.threads.messages.create(
            thread_id=self.thread_id, role="user", content=user_input
        )
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id, assistant_id=self.assistant.id
        )
        run = self.client.beta.threads.runs.retrieve(
            run_id=run.id, thread_id=self.thread_id
        )

        while run.status not in ["completed", "cancelled", "expired", "failed"]:
            self.log.audit(f"Waiting for run to complete. Currently in: {run.status}")
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(
                run_id=run.id, thread_id=self.thread_id
            )
            if run.status == "requires_action":
                self.log.audit("Run requires action.")
                self.process_requires_action(run)

        if run.status == "completed":
            self.log.debug("Run completed.")
            return self.get_response()
        else:
            raise Exception(f"Run ended with status: {run.status}")
