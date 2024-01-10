from openai.types.beta.threads import Run
from openai.types.beta.thread import Thread
from openai.types.beta.assistant import Assistant

import time
import json

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from utils import available_functions

import streamlit as st

from typing import Literal

_: bool = load_dotenv(find_dotenv())

client: OpenAI = OpenAI()


class FinancialAssistantManager:
    def __init__(self, model: str = "gpt-3.5-turbo-1106"):
        self.client = OpenAI()
        self.model = model
        self.assistant: Assistant | None = None
        self.thread: Thread | None = None
        self.run: Run | None = None

    def create_assistant(
        self,
        name: str,
        instructions: str,
        tools: list,
    ) -> Assistant:
        st.sidebar.write("\nCreating Assisatnt")
        if self.assistant is None:
            self.assistant = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                tools=tools,
                model=self.model,
            )
        st.sidebar.write("Assisatnt Created\n")
        return self.assistant

    def create_thread(self) -> Thread:
        st.sidebar.write("\nCreating Thread")
        self.thread = self.client.beta.threads.create()
        st.sidebar.write("Thread Created\n")
        return self.thread

    def add_message_to_thread(self, role: Literal["user"], content: str) -> None:
        st.sidebar.write("\nAdding message to thread")
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id, role=role, content=content
        )
        st.sidebar.write("Message added\n")

    def run_assistant(self, instructions: str) -> Run:
        st.sidebar.write("\nRunning Assisatnt")
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=instructions,
        )
        return self.run

    def wait_for_completion(self, run: Run, thread: Thread):
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id, run_id=self.run.id
            )
            if run.status == "requires_action":
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    if function_name in available_functions:
                        st.sidebar.write("Calling ", function_name)
                        function_to_call = available_functions[function_name]
                        output = function_to_call(**function_args)
                        tool_outputs.append(
                            {
                                "tool_call_id": tool_call.id,
                                "output": output,
                            }
                        )
                self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                )
            elif run.status == "completed":
                processed_response = self.process_messages()
                st.sidebar.write(f"Run is {run.status}.")
                return processed_response

            elif run.status == "failed":
                st.sidebar.write("Run failed.")
                break

            elif run.status in ["in_progress", "queued"]:
                st.sidebar.write(f"Run is {run.status}. Waiting...")
                time.sleep(5)

            else:
                st.sidebar.write(f"Unexpected status: {run.status}")
                break

    def process_messages(self):
        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        return messages
