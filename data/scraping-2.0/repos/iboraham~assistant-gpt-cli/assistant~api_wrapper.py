import time
from halo import Halo
from openai import OpenAI


class AssistantAPIWrapper:
    """
    A wrapper class for the OpenAI API, managing the assistant, threads, and messages.
    """

    def __init__(self, api_key, username, assistant_id=None):
        """
        Initializes the API client and sets up basic parameters.
        """
        self.client = OpenAI(api_key=api_key)
        self.thread = None
        self.assistant = None
        self.run = None
        self.username = username

    def _convert_tools(self, tools):
        """
        Converts a list of tool names into the format required by the OpenAI API.
        """
        return [{"type": tool} for tool in tools]

    def create_assistant(
        self,
        name,
        description=None,
        model="gpt-4-vision-preview",
        instructions=None,
        tools=[],
    ):
        """
        Creates a new assistant with the specified parameters.
        """
        self.assistant = self.client.beta.assistants.create(
            name=name,
            description=description,
            model=model,
            instructions=instructions,
            tools=self._convert_tools(tools),
        )

    def edit_assistant(
        self,
        name,
        description=None,
        model="gpt-4-vision-preview",
        instructions=None,
        tools=[],
    ):
        """
        Edits the existing assistant with new parameters.
        """
        self.assistant = self.client.beta.assistants.update(
            assistant_id=self.assistant.id,
            name=name,
            description=description,
            model=model,
            instructions=instructions,
            tools=self._convert_tools(tools),
        )

    def list_assistants(self):
        """
        Retrieves a list of all assistants.
        """
        return self.client.beta.assistants.list()

    def get_thread(self, thread_id):
        """
        Retrieves a specific thread by its ID.
        """
        return self.client.beta.threads.retrieve(thread_id=thread_id)

    def create_thread(self):
        """
        Creates a new thread and stores it in the instance variable.
        """
        self.thread = self.client.beta.threads.create()

    def add_message_to_thread(self, message, role="user", files=[]):
        """
        Adds a message to the current thread.
        """
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role=role,
            content=message,
            file_ids=files,
        )

    def send_message(self):
        """
        Sends a message via the assistant in the current thread.
        """
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
        )

    def get_messages(self):
        """
        Retrieves all messages from the current thread.
        """
        return self.client.beta.threads.messages.list(thread_id=self.thread.id)

    def check_run_status(self):
        """
        Checks and waits for the run status to complete, with a spinner for user feedback.
        """
        run = self.client.beta.threads.runs.retrieve(
            thread_id=self.thread.id,
            run_id=self.run.id,
        )
        spinner = Halo(text="Thinking...", spinner="dots")
        spinner.start()

        counter = 0
        while run.status in ["in_progress", "queued"]:
            if counter % 10 == 0:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id,
                )
                time.sleep(5)
            counter += 1

        if run.status == "completed":
            spinner.succeed("Done")
        else:
            spinner.fail("Error")
            raise Exception(f"Run failed: {run}")
