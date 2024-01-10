import os
import time
import json
import dotenv
from colors import ANSI_USER_MESSAGE, ANSI_PROBLEM, ANSI_IDENTIFIER, ANSI_ACTION, ANSI_ASSISTANT_MESSAGE, ANSI_RESET
from openai import OpenAI

# Load environment variables
dotenv.load_dotenv()

# Placeholder for assistant ID
assistant_id = "YOUR_ASSISTANT_ID"

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Mapping of tool names to Python function names
tools = {
}


def prettify_json(json_str):
    return json.dumps(json.loads(json_str), indent=2)


def update_run_with_tool_calls(tool_calls, thread, run):
    tool_outputs = []

    # Report how many tool_calls the model is asking for and to which tools
    print(f'{ANSI_ACTION}Model is making {ANSI_IDENTIFIER}{len(tool_calls)}{ANSI_ACTION} tool call(s):{ANSI_RESET}')

    for tool_call in tool_calls:
        if tool_call.type != "function":
            print('Skipping processing for unknown tool type: ' + tool_call.type)

        if tool_call.function.name not in tools:
            pretty_args = prettify_json(tool_call.function.arguments)
            print(f'{ANSI_PROBLEM}Unknown tool {ANSI_IDENTIFIER}{tool_call.function.name}')
            print(f'{ANSI_ACTION}Tool arguments:{ANSI_RESET}')
            print(pretty_args)
            output = input(f"{ANSI_PROBLEM}Please provide tool output:\n> {ANSI_RESET}")
        else:
            print(f'{ANSI_ACTION}Model invoked tool {ANSI_IDENTIFIER}{tool_call.function.name}' +
                  f'{ANSI_ACTION} with args:{ANSI_RESET}')

            print(prettify_json(tool_call.function.arguments))

            tool_func = tools[tool_call.function.name]
            output = tool_func(
                **json.loads(tool_call.function.arguments)
            )
            print(f'{ANSI_ACTION}Tool output:{ANSI_RESET}')
            print(output)

        tool_outputs.append({
            "tool_call_id": tool_call.id,
            "output": output
        })

    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=tool_outputs
    )

    return run


def main():

    # Create a new thread
    thread = client.beta.threads.create()

    while True:
        # Get user input and add it to the thread if it is not empty
        user_input = input(ANSI_USER_MESSAGE + "Enter your message:\n> " + ANSI_RESET)

        if user_input.strip() != "":
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_input
            )

        # Ask the assistant to run on the thread
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )

        # Wait while the agent works on the run
        while run.status not in ["completed", "failed", "cancelled", "expired"]:
            if run.status == "requires_action":
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                run = update_run_with_tool_calls(tool_calls, thread, run)
            else:

                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

            time.sleep(1)

        # Check if we need to submit something for the run to continue

        # Retrieve the latest message
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        ).data
        latest_message = messages[0]
        if latest_message.role == "assistant":
            print(f'{ANSI_ASSISTANT_MESSAGE}Assistant says:{ANSI_RESET}')
            print(latest_message.content[0].text.value)


if __name__ == "__main__":
    main()
