from openai import OpenAI
import time
import json
import snippets.function_tools as func_t

client = OpenAI()

# functions for managing openai assistants

# handling of assistants with "function" tool

# add functions for all assistant "functions" here


# add message to thread
def add_message(message: str, thread_id: str, assistant_id: str):
    """
    Take message and add it to the assistant's thread
    """
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=message
    )
    return


# run thread with instructions
def run_thread(instructions: str, thread_id: str, assistant_id: str):
    """
    Run thread with instructions, returns the run object
    """
    run = client.beta.threads.runs.create(
        thread_id=thread_id, assistant_id=assistant_id, instructions=instructions
    )
    return run


# wait for status
def runstatus_handle(run_id: str, thread_id: str, assistant_id: str):
    """
    Check the status of a run and handle requirements
    """
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

    while run.status != "completed":
        if run.status == "requires_action":
            # find the function and do it
            results = oai_required_action(
                run.required_action.submit_tool_outputs.tool_calls
            )
            # submit results
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id, run_id=run.id, tool_outputs=results
            )
        else:
            # print current status and wait to try again
            print(f"Current status of run: {run.status}")
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

    return run


# required action function
def oai_required_action(calls):
    """
    When 'Require_action' status is given from run, this function handles the calling
    of outside functions to supply data back to the assistant
    calls: openai type of run.required_action.submit_tool_outputs.tool_calls
    returns the 'call_id' and output for each call
    """
    # map for enabled tools in assistant
    tool_functions = {
        "current_time": {"function": func_t.current_time, "params": ["timezone"]}
    }

    results = []
    # for each required call in the required_action object
    # call the right function with the right arguments
    for call in calls:
        # get required function
        func = tool_functions[call.function.name]["function"]
        # get params for the function
        arguments = json.loads(call.function.arguments)
        response = func(**arguments)
        results.append({"tool_call_id": call.id, "output": response})

    return results


# get last message
def get_message_text(thread_id: str, index: int = 0):
    # gets the message at index for a thread (default last message)
    messages = client.beta.threads.messages.list(thread_id=thread_id)

    return messages.data[index].content[0].text.value
