import time
from logging import info

import openai
from openai.types.beta.assistant import Assistant
from openai.types.beta.thread import Thread
from openai.types.beta.assistant_create_params import AssistantCreateParams
from openai.types.beta.threads.run_submit_tool_outputs_params import RunSubmitToolOutputsParams, ToolOutput
from openai.types.beta.threads.required_action_function_tool_call import RequiredActionFunctionToolCall, Function

from .tool_functions import BaseFunction, WriteFileFunction, FunctionEventHandler


class AssistantInterface:
    @classmethod
    def create(cls, client: openai.OpenAI, functions: list[BaseFunction], params: AssistantCreateParams):
        assistant = client.beta.assistants.create(**params)
        info("Created assistant_id '%s'", assistant.id)
        return cls(client, assistant, functions)

    @classmethod
    def retrieve(cls, client: openai.OpenAI, assistant_id: str, functions: list[BaseFunction]):
        assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
        info("Retrieved assistant_id '%s'", assistant.id)
        return cls(client, assistant, functions)

    def __init__(self, client: openai.OpenAI, assistant: Assistant, functions: list[BaseFunction] = []):
        self.client = client
        self.assistant = assistant
        self.functions = {func.NAME: func for func in functions}

    def create_thread(self):
        return ThreadInterface.create(self)

    def retrieve_thread(self, thread_id: str):
        return ThreadInterface.retrieve(self, thread_id)


class ThreadInterface:
    @classmethod
    def create(cls, assistant_interface: AssistantInterface):
        thread = assistant_interface.client.beta.threads.create()
        info("Created thread_id '%s'", thread.id)
        return cls(assistant_interface, thread)

    @classmethod
    def retrieve(cls, assistant_inteface: AssistantInterface, thread_id: str):
        thread = assistant_inteface.client.beta.threads.retrieve(thread_id=thread_id)
        info("Retrieved thread_id '%s'", thread.id)
        return cls(assistant_inteface, thread)

    def __init__(self, assistant_interface: AssistantInterface, thread: Thread):
        self.assistant_interface = assistant_interface
        self.thread = thread
        self.client = self.assistant_interface.client

    @property
    def functions(self):
        return self.assistant_interface.functions

    def send(self, mesg: str):
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=mesg,
        )

        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant_interface.assistant.id,
        )

        return PendingOperation(self, run.id)

    def run_required_action(self, run: openai.types.beta.threads.Run, pending: "PendingOperation"):
        outputs: list[ToolOutput] = []
        action: RequiredActionFunctionToolCall
        for action in run.required_action.submit_tool_outputs.tool_calls:
            info("Should Run: %s: %s", action.function.name, action.function.arguments)

            func_cls = self.functions[action.function.name]
            func = func_cls.model_validate_json(action.function.arguments)

            outputs.append(ToolOutput(output=str(func(pending)), tool_call_id=action.id))

        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=run.id,
            tool_outputs=outputs,
        )


class PendingOperation(FunctionEventHandler):
    def __init__(self, thread_interface, run_id):
        self.thread_interface = thread_interface
        self.run_id = run_id
        self._log_messages: list[str] = []

    def log(self, mesg: str):
        self._log_messages.append(mesg)

    def get_log_messages(self) -> list[str]:
        log_messages = self._log_messages
        if log_messages:
            self._log_messages = []
        return log_messages

    def ready(self):
        run = self.thread_interface.client.beta.threads.runs.retrieve(
            thread_id=self.thread_interface.thread.id,
            run_id=self.run_id,
        )
        match run.status:
            case "completed":
                return True
            case "requires_action":
                self.thread_interface.run_required_action(run, self)
            case "queued" | "in_progress":
                pass
            case _:
                raise ValueError(f"Unhandled status: {run.status}")
        return False

    def get_response(self):
        return list(self.thread_interface.client.beta.threads.messages.list(
            thread_id=self.thread_interface.thread.id,
            limit=1,
        ))[0]
