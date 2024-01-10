from openai import OpenAI
from openai.types.beta.thread import Thread
from openai.types.beta.threads import Run
from openai.types.beta.assistant_create_params import Tool
from typing import List, Dict
from time import sleep
from agent import toolcalls, utils


class Agent:
    # Factory design pattern to prevent duplicated instance
    _instances = {}

    def __init__(
        self,
        client: OpenAI,
        name: str,
        model: str = None,
        instructions: str = None,
        tools: List[Tool] = None,
        file_names: List[str] = None,
    ):
        self.client = client
        self.name = name
        self.file_ids = []

        print(f"Creating Agent {self.name}...")
        # Check if there is an Assistant with the same name. If not, create one.
        self.id = self.client.beta.assistants.create(
            model=model if model else "gpt-3.5-turbo-1106"
        ).id
        
        if instructions:
            self.update_instructions(instructions)
        if tools:
            self.update_tools(tools)
        if file_names:
            self.file_ids = self.update_files(file_names)
            
        self.threads: Dict[str, Thread] = {}

    @classmethod
    def get_or_create(
        cls,
        client: OpenAI,
        name: str,
        model: str = None,
        instructions: str = None,
        tools: List[Tool] = None,
        file_names: List[str] = None,
    ):
        if name not in cls._instances:
            # If the instance doesn't exist, create and store it
            instance = cls(client, name, model, instructions, tools, file_names)
            cls._instances[name] = instance
        else:
            print(f"Returning existing instance of Agent {name}")

        return cls._instances[name]

    def submit_tool_outputs(self, run: Run, thread: Thread):
        if run.status != "requires_action":
            return

        _toolcalls = run.required_action.submit_tool_outputs.tool_calls
        outputs = toolcalls.execute_all(_toolcalls)
        print("Submitting tool outputs...")
        self.client.beta.threads.runs.submit_tool_outputs(
            run_id=run.id, thread_id=thread.id, tool_outputs=outputs
        )

    def process_message(
        self,
        message: str,
        thread_name: str = "new thread",
        file_names: List[str] = [],
    ) -> str:
        # create a new thread if thread name doesn't exist
        thread = (
            self.client.beta.threads.create()
            if thread_name not in self.threads
            else self.threads[thread_name]
        )

        # add thread-specific files
        file_ids = utils.upload_files(self.client, file_names=file_names)
        if file_ids.__len__() != 0:
            self.client.beta.threads.messages.create(
                thread_id=thread.id, file_ids=file_ids, role="user"
            )

        self.client.beta.threads.messages.create(
            thread_id=thread.id, content=message, role="user"
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id, assistant_id=self.id
        )

        # wait for the run to complete
        print(f"{self.name} is processing message...")
        while True:
            run = self.client.beta.threads.runs.retrieve(
                run_id=run.id, thread_id=thread.id
            )
            if run.status == "requires_action":
                self.submit_tool_outputs(run, thread)
            elif run.status == "completed":
                return (
                    self.client.beta.threads.messages.list(thread.id)
                    .data[0]
                    .content[0]
                    .text.value
                )
            elif run.status == "failed":
                print(run.last_error)
                break
            sleep(0.5)

    def update_instructions(self, instructions: str):
        self.client.beta.assistants.update(
            self.id,
            instructions=instructions,
        )

    def update_tools(self, tools: List[Tool]):
        self.client.beta.assistants.update(
            self.id,
            tools=tools,
        )

    def update_files(self, file_names: List[str]):
        self.file_ids = utils.upload_files(self.client, file_names)
        self.client.beta.assistants.update(
            self.id, file_ids=self.file_ids
        )

    def add_files(self, file_names: List[str]):
        self.file_ids.append(utils.upload_files(self.client, file_names))
        self.client.beta.assistants.update(
            self.id, file_ids=self.file_ids
        )