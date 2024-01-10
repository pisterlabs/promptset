import os
import json
import openai
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


# Initialize clients with API keys
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

assistant_prompt_instruction = """You are knowledgeable about every topic.
Your goal is to use internet resources to provide answers.
You should never use
your own knowledge to answer questions.
I will add link to relevant resources at the end of the answer.
"""
# Function to perform a Tavily search
# https://github.com/assafelovic/tavily-python
def tavily_search(query):
    search_result = tavily_client.get_search_context(
        query, search_depth="advanced", max_tokens=8000
    )
    return search_result


# Function to wait for a run to complete
def wait_for_run_completion(thread_id, run_id):
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Current run status: {run.status}")
        if run.status in ["completed", "failed", "requires_action"]:
            return run


# Function to handle tool output submission
def submit_tool_outputs(thread_id, run_id, tools_to_call):
    tool_output_array = []
    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        function_name = tool.function.name
        function_args = tool.function.arguments

        if function_name == "search-news":
            output = tavily_search(query=json.loads(function_args)["query"])

        if output:
            tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id, run_id=run_id, tool_outputs=tool_output_array
    )


# Function to print messages from a thread
def print_messages_from_thread(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for msg in messages:
        print(f"{msg.role}: {msg.content[0].text.value}")
    return messages


def create_assistant():
    # Create an assistant
    assistant = client.beta.assistants.create(
        instructions=assistant_prompt_instruction,
        model="gpt-3.5-turbo-1106",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search-news",
                    "description": "Get information on recent events from the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to use. For example: 'Latest news on Nvidia stock performance'",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
    )
    assistant_id = assistant.id
    print(f"Assistant ID: {assistant_id}")

    # Create a thread
    thread = client.beta.threads.create()
    print(f"Thread: {thread}")
    return assistant_id, thread


def call_assistant(user_input):
    # Create assistant()
    assistant_id, thread = create_assistant()

    # Ongoing conversation loop
    while True:

        # Create a message
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input,
        )

        # Create a run
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )
        print(f"Run ID: {run.id}")

        # Wait for run to complete
        run = wait_for_run_completion(thread.id, run.id)

        if run.status == "failed":
            print(run.error)
            continue
        elif run.status == "requires_action":
            run = submit_tool_outputs(
                thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls
            )
            run = wait_for_run_completion(thread.id, run.id)

        # Print messages from the thread
        return print_messages_from_thread(thread.id)


def generate_chat_completion(user_input):
    messages = call_assistant(user_input)
    return messages
