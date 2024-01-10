import subprocess

from openai import Client
import time
import builtins
import textwrap
from pydantic import Field
from instructor import OpenAISchema
from typing import Literal

from ThreadManager import *
from agent_instructions import *

KeywordGeneratorAgent = client.beta.assistants.create(
    model="gpt-3.5-turbo-1106",
    name="KeywordGeneratorAgent",
    instructions=keyword_generator_instructions
)

SynonymFinderAgent = client.beta.assistants.create(
    model="gpt-3.5-turbo-1106",
    name="SynonymFinderAgent",
    instructions=synonym_finder_instructions
)

SQLQueryWriterAgent = client.beta.assistants.create(
    model="gpt-3.5-turbo-1106",
    name="SQLQueryWriterAgent",
    instructions=sql_query_writer_instructions
)

DataAnalysisAgent = client.beta.assistants.create(
    model="gpt-3.5-turbo-1106",
    name="DataAnalysisAgent",
    instructions=data_analysis_instructions
)


def wprint(*args, width=70, **kwargs):
    wrapper = textwrap.TextWrapper(width=width)
    wrapped_args = [wrapper.fill(str(arg)) for arg in args]
    builtins.print(*wrapped_args, **kwargs)


def get_completion(message, agent, funcs, thread):
    """
    Executes a thread based on a provided message and retrieves the completion result.

    This function submits a message to a specified thread, triggering the execution of an array of functions
    defined within a func parameter. Each function in the array must implement a `run()` method that returns the outputs.

    Parameters:
    - message (str): The input message to be processed.
    - agent (OpenAI Assistant): The agent instance that will process the message.
    - funcs (list): A list of function objects, defined with the instructor library.
    - thread (Thread): The OpenAI Assistants API thread responsible for managing the execution flow.

    Returns:
    - str: The completion output as a string, obtained from the agent following the execution of input message and functions.
    """

    # create new message in the thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message
    )

    # run this thread
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id,
    )

    while True:
        # wait until run completes
        while run.status in ['queued', 'in_progress']:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            time.sleep(1)

        # function execution
        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                wprint('\033[31m' + str(tool_call.function), '\033[0m')
                # find the tool to be executed
                func = next(iter([func for func in funcs if func.__name__ == tool_call.function.name]))

                try:
                    # init tool
                    func = func(**eval(tool_call.function.arguments))
                    # get outputs from the tool
                    output = func.run()
                except Exception as e:
                    output = "Error: " + str(e)

                wprint(f"\033[33m{tool_call.function.name}: ", output, '\033[0m')
                tool_outputs.append({"tool_call_id": tool_call.id, "output": output})

            # submit tool outputs
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
        # error
        elif run.status == "failed":
            raise Exception("Run Failed. Error: ", run.last_error)
        # return assistant message
        else:
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            message = messages.data[0].content[0].text.value
            return message


class KeywordGeneratorFunction(OpenAISchema):
    text: str

    def run(self):
        # Implement your keyword extraction logic here
        # For demonstration, this example simply splits the text into words
        keywords = self.text.split()
        return keywords


class SynonymFinderFunction(OpenAISchema):
    keywords: list[str]

    def run(self):
        # Implement your synonym finding logic here
        # This is a placeholder for demonstration
        synonyms = {keyword: [f"{keyword}_synonym1", f"{keyword}_synonym2"] for keyword in self.keywords}
        return synonyms


class SQLQueryWriterFunction(OpenAISchema):
    keywords: list[str]

    def run(self):
        # Implement your SQL query generation logic here
        # This is a placeholder for demonstration
        queries = [f"SELECT * FROM logs WHERE message LIKE '%{keyword}%'" for keyword in self.keywords]
        return queries


class DataAnalysisFunction(OpenAISchema):
    query_results: list

    def run(self):
        # Implement your data analysis logic here
        # This is a placeholder for demonstration
        analysis = "Data analysis results based on the query results."
        return analysis


agents_and_threads = {
    "KeywordGeneratorAgent": {
        "agent": KeywordGeneratorAgent,
        "thread": None,
        "funcs": [KeywordGeneratorFunction]
    },
    "SynonymFinderAgent": {
        "agent": SynonymFinderAgent,
        "thread": None,
        "funcs": [SynonymFinderFunction]
    },
    "SQLQueryWriterAgent": {
        "agent": SQLQueryWriterAgent,
        "thread": None,
        "funcs": [SQLQueryWriterFunction]
    },
    "DataAnalysisAgent": {
        "agent": DataAnalysisAgent,
        "thread": None,
        "funcs": [DataAnalysisFunction]
    }
}

# Initialize threads for each agent
for agent in agents_and_threads.values():
    agent["thread"] = client.beta.threads.create()


class SendMessage(OpenAISchema):
    recepient: Literal[
        'KeywordGeneratorAgent', 'SynonymFinderAgent', 'SQLQueryWriterAgent', 'DataAnalysisAgent'] = Field(
        ...,
        description="Specify the recipient agent for the message."
    )
    message: str = Field(
        ...,
        description="Specify the task required for the recipient agent to complete."
    )

    def run(self):
        recepient_info = agents_and_threads.get(self.recepient)
        if not recepient_info["thread"]:
            recepient_info["thread"] = client.beta.threads.create()

        message = get_completion(message=self.message, **recepient_info)
        return message


user_proxy = client.beta.assistants.create(
    name='User Proxy Agent',
    instructions=f"""As a user proxy agent, your responsibility is to streamline the dialogue between the user and specialized agents within this group chat.Your duty is to articulate user requests accurately to the relevant agents and maintain ongoing communication with them to guarantee the user's task is carried out to completion.Please do not respond to the user until the task is complete, an error has been reported by the relevant agent, or you are certain of your response.Main Goal is :Assist users in identifying specific issues or information within log files.Focus on accuracy and simple solutions. Your Main task is to pin point the exact rows in the . That are of interest to the user.""",
    model="gpt-3.5-turbo-1106",
    tools=[
        {"type": "function", "function": SendMessage.openai_schema},
    ],
)

## sdsd
thread = client.beta.threads.create()
while True:
    user_message = input("User: ")
    user_proxy_tools = [SendMessage]

    message = get_completion(user_message, user_proxy, user_proxy_tools, thread)

    wprint(f"\033[34m{user_proxy.name}: ", message, '\033[0m')
