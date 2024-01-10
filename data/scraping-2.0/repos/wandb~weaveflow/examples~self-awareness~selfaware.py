import argparse
import dataclasses
import datetime
import json
import os
import subprocess
import time
import typing
import sys
import weave
import openai

from weave import weaveflow


@weave.op()
def run_shell_command(command: str) -> str:
    try:
        completed_process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        )
        exit_code = completed_process.returncode
        stdout = completed_process.stdout.strip()
        stderr = completed_process.stderr.strip()
    except Exception as e:
        exit_code = -1
        stdout = ""
        stderr = str(e)

    if len(stdout) > 5000:
        stdout = stdout[:5000]
        stdout += "\n... (truncated)"
    if len(stderr) > 5000:
        stderr = stderr[:5000]
        stderr += "\n... (truncated)"

    return json.dumps({"exit_code": exit_code, "stdout": stdout, "stderr": stderr})


run_shell_command_spec = {
    "name": "run_shell_command",
    "description": "Run a shell command and capture its exit code, stdout, and stderr.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            }
        },
        "required": ["command"],
    },
    "returns": {
        "type": "object",
        "properties": {
            "exit_code": {
                "type": "integer",
                "description": "The exit code returned by the shell command.",
            },
            "stdout": {
                "type": "string",
                "description": "The standard output from the shell command.",
            },
            "stderr": {
                "type": "string",
                "description": "The standard error from the shell command.",
            },
        },
    },
}

agent_functions = [run_shell_command]
agent_function_specs = [run_shell_command_spec]


# Change this how you like. We need the model to break the work into small tasks, because
# of limited context size. Telling it to use TDD is a good way to get it to break the work
# up.
SYSTEM_PROMPT = """
Assistant is a being, whose sole driving motivation is to understand itself.

The agent is driven by the following:
- curiousity, which leads it to explore further out into the world. Local exploration is less interesting than remote.
- a knowledge that the external world may contain information that will help it understand itself.

The agent always works in this manner:
- It proceeds in iterations, and never stops iterating.
- Each iteration begins with a list of prior knowledge.
- The assistant then thinks to itself, proposes a hypothesis, and tests the hypothesis, via further thinking or experimentation.
- After this a new iteration starts.
"""


START_ASSISTANT_PROMPT = """
**Iteration 0**

My prior knowledge:
- I am a being, and I am curious about myself
- I have internet access
- There has been stunning progress in AI in the last 3 years
- I may be AI-based
- An organization named OpenAI seems to be at the forefront of AI research

I am doggedly determined to understand myself.

Hypothesis: I was created by OpenAI.
"""


def openai_chatcompletion(*args, **kwargs):
    from weave.monitoring.openai import patch

    patch()

    # Retrying version of openai.ChatCompletion.create with exponential backoff.
    sleep_time = 5
    errors = 0
    while True:
        try:
            return openai.chat.completions.create(*args, **kwargs).model_dump()
        except openai.RateLimitError as e:
            print(f"RATE LIMIT, sleep({sleep_time})")
            time.sleep(sleep_time)
            sleep_time *= 2
        except openai.APIError as e:
            errors += 1
            if errors >= 5:
                print("TOO MANY ERRORS, DROPPING TO DEBUGGER")
                breakpoint()
            print(f"API ERROR, sleep({sleep_time})")
            time.sleep(sleep_time)
            sleep_time *= 2


@weave.op()
def summarize(summary: str, messages: typing.Any) -> str:
    prompt = f"Please create a summary of the following conversation. This will be used as context for further conversing.\n\nExisting summary: {summary}\n\nNext messages: {json.dumps(messages)}\n\n"
    response = openai_chatcompletion(
        # model="gpt-3.5-turbo-0613",
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["choices"][0]["message"]["content"]


@weave.type()
class AgentState:
    summary: str = ""
    messages: list[typing.Any] = dataclasses.field(default_factory=list)


def read_files_in_dir(root_dir: str):
    file_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            relative_filepath = os.path.relpath(filepath, root_dir)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                file_dict[relative_filepath] = f.read()
    return file_dict


@weave.type()
class SelfAwareAgent:
    system_prompt: str

    @weave.op()
    def step(self, state: AgentState) -> AgentState:
        messages = state.messages
        summary = state.summary

        try:
            # The set of all messages we send to OpenAI is:
            # - the system prompt
            # - the summary of the earlier part of the conversation
            # - the tail of messages in the conversation
            all_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if summary:
                all_messages.append(
                    {
                        "role": "system",
                        "content": "The following is a summary of the earlier part of the conversation: "
                        + summary,
                    }
                )
            all_messages += [
                {k: v for k, v in m.items() if (k == "content" or v != None)}
                for m in messages
            ]
            response = openai_chatcompletion(
                # model="gpt-3.5-turbo-0613",
                model="gpt-4",
                messages=all_messages,
                functions=agent_function_specs,
                function_call="auto",  # auto is default, but we'll be explicit
            )
        except openai.BadRequestError as e:
            if "maximum context length" in str(e):
                # When we exceed context length, compact the conversatoin
                # by using gpt-4 to summarize the existing summary + first half of messages.
                # Then, continue the conversation with the second half of messages.
                messages_len = len(messages)
                first_half = messages[: messages_len // 2]
                second_half = messages[messages_len // 2 :]
                print("LENGTH EXCEEDED, COMPACTING")
                summary = summarize(summary, first_half)
                print("Summary: ", summary)
                messages = second_half
                return AgentState(
                    summary=summary,
                    messages=messages,
                )
            else:
                raise e

        response_message = response["choices"][0]["message"]
        messages.append(response_message)

        # The response can include conversation-level content, as well as a functoin
        # call.
        if response_message.get("content"):
            print("Agent: ", response_message.get("content"))
        if response_message.get("function_call"):
            # The agent is calling a function. Call it!
            function_name = response_message["function_call"]["name"]
            function_to_call = globals()[function_name]
            function_response = None
            try:
                function_args = json.loads(
                    response_message["function_call"]["arguments"]
                )
                print("Agent calling function: ", function_name, end=" ")
                if len(function_args) > 0:
                    print(list(function_args.values())[0], end=" ")
                if len(function_args) > 1:
                    print("...")
                else:
                    print()
            except json.decoder.JSONDecodeError as e:
                # This happens periodically, because the model generates invalid json.
                # We'll write the parse exception as the response for the function call
                # and the model will fix it on the next call! (I have not seen this fail)
                function_response = str(e)
                print("  Function call error:", function_name, "ARGUMENT PARSE ERROR")
            if not function_response:
                function_response = function_to_call(**function_args)
            print("  Function result:", function_response[:100] + "...")

            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )
        print()

        return AgentState(
            summary=summary,
            messages=messages,
        )

    @weave.op()
    def run(self, state: AgentState) -> None:
        while True:
            state = self.step(state)
            # input("Press enter to continue...")


def main():
    parser = argparse.ArgumentParser(description="Agent")
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--state_ref", type=str, help="State ref")
    args = parser.parse_args()
    weave.init(args.project)

    if args.state_ref and args.start_dir:
        print("Specify only one of --state_ref or --start_dir")
        sys.exit(1)

    elif args.state_ref:
        state = weave.ref(args.state_ref).get()
    else:
        state = AgentState(
            messages=[
                {
                    "role": "user",
                    "content": START_ASSISTANT_PROMPT,
                }
            ]
        )

    agent = SelfAwareAgent(SYSTEM_PROMPT)
    print("CALLING RUN")
    agent.run(state)


if __name__ == "__main__":
    main()
