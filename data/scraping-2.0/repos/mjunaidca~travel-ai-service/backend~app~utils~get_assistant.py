from openai.types.beta.assistant import Assistant
from openai import OpenAI

from ..utils.seed_assistant import travel_agent_tools, SEED_INSTRUCTION


from openai.types.beta.assistant import Assistant
from openai import OpenAI


class GetAssistant():
    def __init__(self, client: OpenAI):
        if client is None:
            raise Exception("OpenAI Client is not initialized")
        self.client = client
        print("OpenAI Client initialized successfully.")

    def retrieve_assistant(self, assistant_id: str) -> Assistant:
        """Retrieve an Assistant using the ID stored in the env variables.
           If the assistant is not found, create a new one."""
        print(f"Attempting to retrieve Assistant with ID: {assistant_id}")

        if assistant_id is None or assistant_id == '':
            print("No valid Assistant ID provided. Creating a new Assistant.")
            travel_agent = self.create_assistant()
            return travel_agent

        try:
            print(f"Retrieving existing Assistant with ID: {assistant_id}")
            ret_travel_agent: Assistant = self.client.beta.assistants.retrieve(
                assistant_id=assistant_id
            )
            print("Assistant retrieved successfully.")
            return ret_travel_agent

        except Exception as e:
            print(f"""Error retrieving Assistant: {
                  e}. Creating a new Assistant.""")
            travel_agent = self.create_assistant()
            return travel_agent

    def create_assistant(self) -> Assistant:
        """Create an Assistant Once and Store its ID in the env variables.
           Next retrieve the assistant and use it. You can modify it."""
        print("Creating a new Assistant...")

        travel_agent: Assistant = self.client.beta.assistants.create(
            model="gpt-4-1106-preview",
            name="AI Travel Agent",
            instructions=SEED_INSTRUCTION,
            tools=travel_agent_tools
        )

        print("New Assistant created successfully.")
        return travel_agent
