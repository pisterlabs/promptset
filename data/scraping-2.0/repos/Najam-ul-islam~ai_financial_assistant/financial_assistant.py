from typing import Any
from openai import OpenAI
import streamlit as st
import time

# ===============================================================
# Financial Assistant Class
# ===============================================================
class FinancialAssistant:
    """Financial Assistant that helps with financial questions."""
    def __init__(self, apikey:str) -> None:
        """Initializes the financial assistant."""
        self.client: Any = self.initialize_openai_client(apikey=apikey)
        self.assistant = self.create_assistant()
    # ===============================================================
    # ===============================================================
    # Initialize OpenAI Client      STEP 1
    # ===============================================================
    def initialize_openai_client(self, apikey: str) -> Any:
        """Initializes OpenAI client with the given API key."""
        openai_client: Any = OpenAI(api_key=apikey)
        return openai_client

    # ===============================================================
    # define Assistant    STEP 2
    # ===============================================================
    def create_assistant(self) -> Any:
        """Assistant that helps with financial questions"""
        asst_response: Any = self.client.beta.assistants.create(
            name="Financial Assistant",
            instructions="You are a helpful  financial analyst expert and, focusing on management discussions and financial results. help people learn about financial needs and guid them towards fincial literacy.",
            tools=[{"type": "code_interpreter"}, {"type": "retrieval"}],
            model="gpt-3.5-turbo-1106",
        )
        return asst_response

    # ===============================================================
    # Show Assistant Info
    # ===============================================================
    def show_assistant_info(self, assistant_obj: Any = None) -> None:
        """Shows assistant information in JSON format."""
        match assistant_obj:
            case None:
                assistant_obj = self.assistant
            case _:
                pass
        # json_obj = json.dumps(json.loads(assistant_obj.model_dump_json()), indent=4)
        st.json(assistant_obj.model_dump_json())

    # ===============================================================
    # Define create thread     STEP 3
    # ===============================================================
    def create_thread(self) -> Any:
        """Creates a thread for the assistant."""
        my_thread = self.client.beta.threads.create()
        return my_thread

    # ===============================================================
    # Managing Messages and Running the Assistant   STEP 4
    ## 4.1 Adding Messages to the Thread
    ## 4.2 Running the Assistant on the Thread
    # ===============================================================
    # add a message to the thread and run the assistant on the thread
    def submit_message(self, assistant_id: Any, thread: Any, user_message: str) -> Any:
        """Adds a message to the thread and runs the assistant on the thread."""
        self.client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
        )
        return self.client.beta.threads.runs.create(thread_id=thread.id,assistant_id=assistant_id,)
    # continuously track  the status of run and wait for the assistant to finish running
    def wait_on_run(self, run: Any, thread: Any) -> Any:
        """Waits for the assistant to finish running."""
        while run.status == "queued" or run.status == "in_progress":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run
    # fetch a response from a thread after the completion of a run
    def get_response(self, thread: Any) -> Any:
        """Fetches the response with list of messages from the thread."""
        # get the list of messages from the thread
        messages: Any = self.client.beta.threads.messages.list(
            thread_id=thread.id,
            order="asc",
        )
        return messages
    # Present the response to the user in a readable format
    def pretty_print_response(self, messages: Any) -> Any:
        """Prints the response in a readable format."""
        responses: Any = []
        for msg in messages:
            if msg.role == "assistant":
                # responses.append(msg.content[0].text.value)
                responses.append(msg.content[0].text.value)
        return '\n'.join(responses)
# ===============================================================
# if __name__ == "__main__":
#     financial_assistant = FinancialAssistant(apikey="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")
#     my_thread: Any = financial_assistant.create_thread()
#     submitmessage: Any = financial_assistant.submit_message(assistant_id=financial_assistant.assistant.id, thread=my_thread, user_message="I need help with my finances")
#     run_thread: Any = financial_assistant.wait_on_run(run=submitmessage, thread=my_thread)
#     response: Any = financial_assistant.get_response(thread=my_thread)
#     st.text_area(financial_assistant.pretty_print_response(messages=response), height=400,placeholder="Response")