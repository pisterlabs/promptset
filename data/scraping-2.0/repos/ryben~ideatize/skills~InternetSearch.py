import json
import os
import time

from openai import OpenAI
from openai.types.beta.threads import Run
from tavily import TavilyClient

from skills.BaseTask import BaseTask


class InternetSearch(BaseTask):
    run: Run

    def __init__(self):
        # Initialize clients with API keys
        super().__init__()
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        self.assistant_prompt_instruction = """You are a researcher.
        You search the internet for current trends to gather ideas on what apps to create.
        """

    def execute(self, inputs):
        # Create an assistant
        assistant = self.client.beta.assistants.create(
            instructions=self.assistant_prompt_instruction,
            model="gpt-4-1106-preview",
            tools=[{
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "description": "Get information on recent events from the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string",
                                      "description": "The search query to use. For example: 'What's trending right now in philippines'"},
                        },
                        "required": ["query"]
                    }
                }
            }]
        )
        assistant_id = assistant.id
        print(f"Assistant ID: {assistant_id}")

        # Create a thread
        thread = self.client.beta.threads.create()
        print(f"Thread: {thread}")

        user_input = self.details
        print(f"Instructions: {user_input}")

        # Create a message
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input,
        )

        # Create a run
        self.run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )
        print(f"Run ID: {self.run.id}")

        # Wait for run to complete
        self.run = self.wait_for_run_completion(thread.id, self.run.id)

        if self.run.status == 'failed':
            print(self.run.error)
        elif self.run.status == 'requires_action':
            self.run = self.submit_tool_outputs(thread.id, self.run.id,
                                                self.run.required_action.submit_tool_outputs.tool_calls)
            self.run = self.wait_for_run_completion(thread.id, self.run.id)

        # Print messages from the thread
        # self.print_messages_from_thread(thread.id)

        return self.get_messages_from_thread(thread.id)

    # Function to perform a Tavily search
    def tavily_search(self, query):
        search_result = self.tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)
        return search_result

    # Function to wait for a run to complete
    def wait_for_run_completion(self, thread_id, run_id):
        prev_run_status = ""

        while True:
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

            if prev_run_status != run.status:
                prev_run_status = run.status
                print(f"Run status: {prev_run_status} ", end=" ")
            else:
                print(f".", end="")

            if run.status in ['completed', 'failed', 'requires_action']:
                return run

    # Function to handle tool output submission
    def submit_tool_outputs(self, thread_id, run_id, tools_to_call):
        tool_output_array = []
        for tool in tools_to_call:
            output = None
            tool_call_id = tool.id
            # function_name = tool.function.alias
            function_args = tool.function.arguments

            # if function_name == "tavily_search":
            output = self.tavily_search(query=json.loads(function_args)["query"])

            if output:
                tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

        return self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_output_array
        )

    # Function to print messages from a thread
    def print_messages_from_thread(self, thread_id):
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        for msg in messages:
            print(f"{msg.role}: {msg.content[0].text.value}")

    # Function to print messages from a thread
    def get_messages_from_thread(self, thread_id) -> str:
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        output = ""
        for msg in messages:
            output = f"{output}\n{msg.role}: {msg.content[0].text.value}"
        return output
