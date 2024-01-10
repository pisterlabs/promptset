# A Class to Manage All Open API Assistant Calls and Functions
from openai.types.beta.threads import Run
from openai.types.beta.thread import Thread
from openai.types.beta.assistant import Assistant
from openai.types.beta.assistant_create_params import Tool

import time
import json

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from seed import available_functions

import streamlit as st

from typing import Literal

_: bool = load_dotenv(find_dotenv())  # read local .env file

client: OpenAI = OpenAI()

class FinancialAssistantManager:
    def __init__(self, model: str = "gpt-3.5-turbo-1106"):
        self.client = OpenAI()
        self.model = model
        self.assistant: Assistant | None = None
        self.thread: Thread | None = None
        self.run: Run | None = None

    def list_assistants(self) -> list:
        """Retrieve a list of assistants."""
        assistants_list = self.client.beta.assistants.list()
        assistants = assistants_list.model_dump()
        return assistants['data'] if 'data' in assistants else []

    def modifyAssistant(self, assistant_id: str, new_instructions: str, tools: list, file_obj: list[str]) -> Assistant:
        """Update an existing assistant."""
        print("Updating edisting assistant...")
        self.assistant = self.client.beta.assistants.update(
            assistant_id=assistant_id,
            instructions=new_instructions,
            tools=tools,
            model=self.model,
            file_ids=file_obj
        )
        return self.assistant

    def find_and_set_assistant_by_name(self, name: str, instructions: str, tools: list[Tool], file_obj: list[str]) -> None:
        """Find an assistant by name and set it if found."""
        assistants = self.list_assistants()
        print("Retrieved assistants list...")
        if self.assistant is None:  # Check if assistant is not already set
            for assistant in assistants:
                if assistant['name'] == name:
                    print("Found assistant...",  assistant['name'] == name)
                    print("Existing Assitant ID", assistant['id'])
                    # self.assistant = assistant
                    self.modifyAssistant(
                        assistant_id=assistant['id'],
                        new_instructions=instructions,
                        tools=tools,
                        file_obj=file_obj
                    )
                    break

    def create_assistant(self, name: str, instructions: str, tools: list, file_obj: list[str], model: str = "gpt-3.5-turbo-1106") -> Assistant:
        """Create or find an assistant."""
        self.find_and_set_assistant_by_name(
            name=name,
            instructions=instructions,
            tools=tools,
            file_obj=file_obj)
        if self.assistant is None:  # Check if assistant is not already set
            print("Creating new assistant...")
            self.assistant = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                tools=tools,
                model=model,
                file_ids=file_obj
            )
        return self.assistant  # Return the assistant object

    def create_thread(self) -> Thread:
        self.thread = self.client.beta.threads.create()
        return self.thread

    def add_message_to_thread(self, role: Literal['user'], content: str) -> None:
        if self.thread is None:
            raise ValueError("Thread is not set!")

        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role=role,
            content=content
        )

    def run_assistant(self, instructions: str) -> Run:

        if self.assistant is None:
            raise ValueError(
                "Assistant is not set. Cannot run assistant without an assistant.")

        if self.thread is None:
            raise ValueError(
                "Thread is not set!")

        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=instructions
        )
        return self.run

    def wait_for_completion(self, run: Run, thread: Thread):

        if self.run is None:
            raise ValueError("Run is not set!")

        # while run.status in ["in_progress", "queued"]:
        while run.status not in ["completed", "failed"]:
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=self.run.id
            )
            time.sleep(3)  # Wait for 3 seconds before checking again
            st.sidebar.write(f"Status: {run_status.status}")

            if run_status.status == 'completed':
                processed_response = self.process_messages()
                return processed_response
                # break
            elif run_status.status == 'requires_action' and run_status.required_action is not None:
                print("Function Calling ...")
                st.sidebar.write(f"Function Calling ...")
                self.call_required_functions(
                    run_status.required_action.submit_tool_outputs.model_dump())
            elif run.status == "failed":
                print("Run failed.")
                break
            else:
                print(f"Waiting for the Assistant to process...: {run.status}")

                # st.sidebar.write(f"Status: {run_status.status}")

    def process_messages(self):
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id)
        return messages

    def call_required_functions(self, required_actions):
        tool_outputs = []

        for action in required_actions["tool_calls"]:
            function_name = action['function']['name']
            arguments = json.loads(action['function']['arguments'])
            print('function_name', function_name)
            print('function_arguments', arguments)

            # Displaying the message with values
            st.sidebar.write(f"Calling {function_name} with:")
            for key, value in arguments.items():
                st.sidebar.write(f"{key}: {value}")

            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                output = function_to_call(**arguments)
                st.sidebar.write("Success...")
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": output,
                })

            else:
                st.sidebar.write(f"Unknown function: {function_name}")
                st.stop()
                raise ValueError(f"Unknown function: {function_name}")

        print("Submitting outputs back to the Assistant...")
        st.sidebar.write("Submitting outputs back to the Assistant...")
        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=self.run.id,
            tool_outputs=tool_outputs
        )
