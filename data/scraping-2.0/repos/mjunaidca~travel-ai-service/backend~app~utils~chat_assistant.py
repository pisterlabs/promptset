from typing import Literal, Callable
from openai.types.beta.threads import Run
from openai.types.beta.thread import Thread
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput

from openai import OpenAI

from ..utils.seed_assistant import available_functions

import time
import json


class TravelAIChat():
    def __init__(self, client: OpenAI, assistant: Assistant, thread: Thread):
        if (client is None):
            raise Exception("OpenAI Client is not initialized")
        self.client = client
        self.assistant: Assistant | None = assistant
        self.thread: Thread | None = thread

    def modifyAssistant(self, new_instructions: str, tools: list, file_obj: list[str], model: str = "gpt-4-1106-preview") -> Assistant:
        """Update an existing assistant."""
        print("Updating edisting assistant...")
        if self.assistant is None:
            raise ValueError("Assistant is not set!")
        self.assistant = self.client.beta.assistants.update(
            assistant_id=self.assistant.id,
            instructions=new_instructions,
            tools=tools,
            model=model,
            file_ids=file_obj
        )
        return self.assistant

    def add_message_to_thread(self, role: Literal['user'], content: str, file_obj_ids: list[str] = ['']) -> None:
        if self.thread is None:
            raise ValueError("Thread is not set!")

        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role=role,
            content=content,
            file_ids=file_obj_ids
        )

    def run_assistant(self) -> Run:

        if self.assistant is None:
            raise ValueError(
                "Assistant is not set. Cannot run assistant without an assistant.")

        if self.thread is None:
            raise ValueError(
                "Thread is not set!")

        self.run: Run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
        )
        return self.run

    # Polling
    def wait_for_completion(self, run: Run):

        if run is None:
            raise ValueError("Run is not set!")

        if self.thread is None:
            raise ValueError(
                "Thread is not set!")

        # while run.status in ["in_progress", "queued"]:
        while run.status not in ["completed", "failed"]:
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )
            time.sleep(3)  # Wait for 3 seconds before checking again
            print(f"Status: {run_status.status}")

            if run_status.status == 'completed':
                print("Run completed.")
                return self.client.beta.threads.messages.list(thread_id=self.thread.id)
            elif run_status.status == 'requires_action' and run_status.required_action is not None:
                print(f"Function Calling ...")
                toolCalls = run_status.required_action.submit_tool_outputs.model_dump()
                self.call_required_functions(
                    toolCalls=toolCalls,
                    thread_id=self.thread.id,
                    run_id=run.id
                )
            elif run.status == "failed":
                print("Run failed.")
                break
            else:
                print(f"Waiting for the Assistant to process...: {run.status}")

    # Function to call the required functions
    def call_required_functions(self, toolCalls, thread_id: str, run_id: str):

        tool_outputs: list[ToolOutput] = []

        # for toolcall in toolCalls:
        for toolcall in toolCalls["tool_calls"]:
            function_name = toolcall['function']['name']
            function_args = json.loads(toolcall['function']['arguments'])

            if function_name in available_functions:

                # Displaying the message with values
                print(f"calling function {function_name} with args:")
                for key, value in function_args.items():
                    print(f"{key}: {value}")

                if function_name in available_functions:
                    function_to_call: Callable[...,
                                               dict] = available_functions[function_name]
                    print("function_to_call >>>>>", function_to_call)
                    output = function_to_call(**function_args)

                    print("Output Status", output)

                    tool_outputs.append({
                        "tool_call_id": toolcall['id'],
                        "output": output['status'] if 'status' in output else output,
                    })

            else:
                raise ValueError(f"Unknown function: {function_name}")

        print('submit_tool_outputs >>>>>', tool_outputs,)
        # Submit tool outputs and update the run
        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs
        )
